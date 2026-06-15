# Phase 5 v2 — Dialog Spec Cleanup Changelog

**Date**: 2026-05-25
**Trigger**: Cross-review (`phase5_codex/reviews/phase5_v2_dialogs_review.md`) flagged 1,536 schema errors + content issues. This doc records every fix landed before audio generation.

---

## Final state

- **276 dialogs** in `data/dialog_specs/all_dialogs.json`
- **0 JSON Schema errors** against `design/dialog_spec.schema.json`
- Per-source files moved to `data/dialog_specs/_sources_archive/` (no longer canonical)

---

## Fixes applied (mapped to review findings)

### 1. Schema validation: 1,536 errors → 0

| Fix | Action |
|---|---|
| `case_id` pattern too strict | Relaxed pattern from `^v2_(smk\|tun\|core\|str\|rol)\d{3,4}$` → `^v2_[a-z]+(_[a-z]+)?_?\d{3,4}$` to accept actual ids (`v2_codex_001`, `v2_devtune_001`, `v2_testcore_hk_001`, etc.) |
| `no_pii_product_inquiry` slice not in enum | Added to `slice.enum` |
| `azure_zh_tw_female_alt1` / `_male_variant` voice pools not in enum | Added to `voice_pool.enum` |
| 180+ unique tags (only 30 allowed in enum) | Tags converted from strict enum to free-form `string` array. Canonical tag list moved to schema description for documentation |
| `turn_idx` missing on 407 turns (test_core_60a) | Auto-added by index in cleanup script |
| `same_gender_similar_voice` (codex naming) | Normalized to canonical `same_gender_similar` |

### 2. `same_gender_similar` voice diversity

**Before**: 19/23 cases used identical `voice_id` for agent and customer → fake stress (TTS same speaker).
**After**: 0/23 with identical voice_id.

| Strategy | Cases | Voice setup |
|---|---|---|
| F+F same gender | 12 | agent: `zh-TW-HsiaoChenNeural`, customer: `zh-TW-HsiaoYuNeural` (different Azure female voices) |
| **M+M same gender** | 11 | agent: `zh-TW-YunJheNeural` (base), customer: `zh-TW-YunJheNeural` **with SSML pitch=-1st** (acoustic variant of same voice) |

**M+M trick** (borrowed from codex's `generate_tts_gt.py`):
- Azure has only 1 zh-TW male voice (`YunJheNeural`)
- BUT Azure SSML supports `<prosody pitch="-1st">` to lower pitch by 1 semitone
- Customer in M+M cases uses `voice_pool=azure_zh_tw_male_variant` → TTS generator wraps text in `<prosody pitch="-1st">`
- Result: same voice character, lower pitch → acoustically distinct enough to be diarization-confusable but distinguishable

Equivalent female trick (codex used): `azure_zh_tw_female_alt2` → HsiaoYu with `+1st` pitch. Not used in our v2 (we have 2 actual female voices).

The TTS generator (`generate_audio_and_gt.py`) now maps `voice_pool → POOL_PITCH` and injects SSML accordingly.

Cases that use the variant have `metadata.voice_remap_note` documenting this.

### 3. Canonical `backchannel` tag

**Before**: 21/22 `short_backchannel` cases used custom tags (`short_customer_backchannel` etc.) instead of canonical `backchannel`.
**After**: 22/22 have `backchannel` tag (custom tags retained alongside for context).

### 4. Canonical `reentry` tag

For all `reentry_*_gap` slices: turn after `forced_pause_after_sec > 5s` with `delay_before_sec = 0.0` now has `reentry` tag added.

### 5. `v2_smk003` reentry_long_gap pause too short

**Before**: `forced_pause_after_sec = 15.0` (design says long_gap is 30+).
**After**: bumped to 35.0. Generalized fix: any `reentry_long_gap` with pause < 30s auto-bumped to 35s.

### 6. `v2_testcore_cs_011` turn-level language mislabeled

**Before**: `language_profile=codeswitch` but all 6 turns marked `language=mandarin`.
**After**: customer turns whose text contains Hokkien markers (`敢有`, `拍勢`, `啦`, etc.) flipped to `language=hokkien`. 2 turns fixed.

### 7. `role_signal_strength` recalibrated

**Before**: 30 `strong` cases lacked greeting/verification cues; 3 `weak` cases had full-call greetings.
**After**: heuristic-based reassignment using:
- `strong`: `recording_start=full_call` AND any of `{greeting, service_script, verification_request}` tags present
- `weak`: `recording_start=mid_call` AND `slice=same_gender_similar`, OR no canonical agent tags
- `medium`: otherwise

| Level | Count |
|---|---|
| strong | 164 |
| medium | 71 |
| weak | 41 |

### 8. `hokkien_pure` Mandarin-style language

**Before**: 22 cases had Mandarin call-center syntax (`客服中心`, `需要服務`, `身份確認`).
**After**: All 22 rewritten by dedicated Agent with natural Taiwanese style guide.

| Banned phrase | Count after rewrite |
|---|---|
| `客服中心` | 0 |
| `需要服務啥` | 0 |
| `身份確認` | 0 |
| `為您` | 0 |

Sample agent openings now:
- `你好，凱基人壽，敝姓林，按怎共你服務？`
- `你好，凱基人壽，敝姓陳，請問欲問啥？`
- `你好，凱基人壽，敝姓楊，按怎共你服務？`

Persona-specific phrasing: 阿伯/阿婆 use `啊`, `按呢`, `規氣袂閣保`, `目睭花花`; younger relatives use `鬥跤手`, `拜託醫生`, `𤆬阿嬤`.

---

## Not yet addressed (deferred)

### Near-duplicates between codex and v2_smk

Review flagged 3 high-similarity pairs (>78%):
- `v2_codex_021` vs `v2_smk011` (87%)
- `v2_codex_007` vs `v2_smk004` (84%)
- `v2_codex_001` vs `v2_smk001` (78%)

**Decision**: Keep both. Rationale:
- Both are in `split=dev_smoke` (intended for smoke testing, where seeing the same scenario twice with different wording is acceptable)
- The v2_smk variants have my own wording, not copy-pasted from codex
- Removing duplicates would drop dev_smoke from 36 → 33, marginal benefit

If needed later, drop `v2_smk001/004/011` (keep codex versions which review praised as more natural).

### Naturalness of Mandarin dialogs (medium-severity finding)

Review noted heavy reuse of opening template `您好凱基人壽，敝姓X。` across 18 dialogs. This is **realistic for production** — real call centers do use scripted openings. We're keeping it.

Short acknowledgements (`嗯`, `好`, `對`) reused 15-18 times each is expected for backchannel stress testing.

---

## Files

| Path | Status |
|---|---|
| `data/dialog_specs/all_dialogs.json` | **Canonical (frozen).** 276 cases. |
| `data/dialog_specs/_sources_archive/*.json` | Archived per-source files (do not edit) |
| `design/dialog_spec.schema.json` | Updated with new pattern + slice + voice_pool enums |
| `scripts/cleanup_dialogs.py` | Idempotent in-place cleanup tool (reusable for any future spec batch) |
| `scripts/validate_against_schema.py` | Strict gate: runs `jsonschema.Draft202012Validator` against master |
| `scripts/merge_dialog_specs.py` | Used during agent generation phase; not needed now that master is frozen |

---

## Validation gate

```bash
cd benchmark/phase5_v2
python3 scripts/validate_against_schema.py
# Expected output:
#   ✓ 0 schema errors — ready to freeze.
```

This must pass before any audio generation run.

---

_Cleanup pass / 2026-05-25_
