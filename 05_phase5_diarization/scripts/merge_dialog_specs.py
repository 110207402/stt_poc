"""
Merge + validate all dialog spec files in data/dialog_specs/ into one master file.

Validates against the v2 schema, drops duplicates by case_id, sorts deterministically.

Usage:
    python scripts/merge_dialog_specs.py
"""

from __future__ import annotations
import json
import pathlib
import sys
import collections

V2_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS_DIR = V2_ROOT / "data" / "dialog_specs"
MASTER_OUT = SPECS_DIR / "all_dialogs.json"

# Files to ignore (sample / scratch)
SKIP_FILES = {"dev_smoke_sample.json", "all_dialogs.json"}

# Required top-level fields
REQUIRED_TOP = {"case_id", "split", "slice", "language_profile", "participants", "turns"}
REQUIRED_PARTICIPANT = {"role", "gender", "voice_id"}
REQUIRED_TURN = {"speaker", "language", "text"}

# Allowed slice names (canonical v2 names)
VALID_SLICES = {
    "clean_turn_taking", "reentry_short_gap", "reentry_long_gap",
    "agent_barge_in", "customer_barge_in", "simultaneous_start",
    "short_backchannel", "hokkien_pure", "codeswitch",
    "same_gender_similar", "third_party_background", "no_pii_product_inquiry",
}

# Normalize codex/legacy slice names → canonical v2 names
SLICE_ALIASES = {
    "hokkien": "hokkien_pure",
    "same_gender_similar_voice": "same_gender_similar",
}

# Allowed voice ids
VALID_VOICES = {
    "zh-TW-HsiaoChenNeural", "zh-TW-HsiaoYuNeural", "zh-TW-YunJheNeural",
    "tai_female_1", "tai_male_1",
}


def validate_case(c: dict, src: str) -> list[str]:
    """Return list of validation errors for one case."""
    errs = []
    case_id = c.get("case_id", "<no_id>")

    # Top-level fields
    missing = REQUIRED_TOP - set(c.keys())
    if missing:
        errs.append(f"{case_id}: missing top-level fields {missing}")

    # Slice
    if c.get("slice") not in VALID_SLICES:
        errs.append(f"{case_id}: invalid slice '{c.get('slice')}'")

    # Participants
    parts = c.get("participants", {})
    if not isinstance(parts, dict) or "agent" not in parts or "customer" not in parts:
        errs.append(f"{case_id}: participants must include agent + customer")
    else:
        for role_key, p in parts.items():
            mp = REQUIRED_PARTICIPANT - set(p.keys())
            if mp:
                errs.append(f"{case_id}.{role_key}: missing {mp}")
            if p.get("voice_id") not in VALID_VOICES:
                errs.append(f"{case_id}.{role_key}: invalid voice_id '{p.get('voice_id')}'")

    # Turns
    turns = c.get("turns", [])
    if not turns:
        errs.append(f"{case_id}: empty turns")
    for i, t in enumerate(turns):
        mt = REQUIRED_TURN - set(t.keys())
        if mt:
            errs.append(f"{case_id}.turn[{i}]: missing {mt}")
        if t.get("speaker") not in ("agent", "customer", "other"):
            errs.append(f"{case_id}.turn[{i}]: invalid speaker '{t.get('speaker')}'")

    return errs


def main():
    all_cases = []
    seen_ids = set()
    all_errors = []
    file_stats = collections.OrderedDict()

    print(f"Scanning {SPECS_DIR}...\n")

    spec_files = sorted(SPECS_DIR.glob("*.json"))
    for sf in spec_files:
        if sf.name in SKIP_FILES:
            continue

        try:
            cases = json.loads(sf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            all_errors.append(f"{sf.name}: JSON parse error: {e}")
            file_stats[sf.name] = {"loaded": 0, "errors": 1, "added": 0, "duplicates": 0}
            continue

        if not isinstance(cases, list):
            all_errors.append(f"{sf.name}: not a JSON array")
            continue

        loaded = len(cases)
        errors_this_file = 0
        added = 0
        duplicates = 0

        for c in cases:
            # Normalize legacy slice names before validation
            if c.get("slice") in SLICE_ALIASES:
                c["slice"] = SLICE_ALIASES[c["slice"]]
            errs = validate_case(c, sf.name)
            if errs:
                all_errors.extend([f"  {sf.name} → {e}" for e in errs])
                errors_this_file += 1
                continue

            cid = c["case_id"]
            if cid in seen_ids:
                duplicates += 1
                print(f"  ! Duplicate case_id {cid} in {sf.name} — skipping")
                continue
            seen_ids.add(cid)

            # Annotate with source file
            c.setdefault("metadata", {})["source_file"] = sf.name
            all_cases.append(c)
            added += 1

        file_stats[sf.name] = {
            "loaded": loaded, "errors": errors_this_file,
            "added": added, "duplicates": duplicates,
        }

    # Print summary
    print(f"{'File':40} {'Loaded':>8} {'Errors':>8} {'Added':>8} {'Dups':>6}")
    print("-" * 75)
    for fname, stats in file_stats.items():
        print(f"{fname:40} {stats['loaded']:>8} {stats['errors']:>8} {stats['added']:>8} {stats['duplicates']:>6}")
    print("-" * 75)
    print(f"{'TOTAL':40} {sum(s['loaded'] for s in file_stats.values()):>8} {sum(s['errors'] for s in file_stats.values()):>8} {len(all_cases):>8}")

    if all_errors:
        print(f"\n{len(all_errors)} validation errors:")
        for e in all_errors[:30]:
            print(f"  {e}")
        if len(all_errors) > 30:
            print(f"  ... and {len(all_errors) - 30} more")

    print()
    print("Slice distribution (merged):")
    for s, n in sorted(collections.Counter(c["slice"] for c in all_cases).items()):
        print(f"  {s:30} {n}")
    print()
    print("Language distribution (merged):")
    for l, n in sorted(collections.Counter(c["language_profile"] for c in all_cases).items()):
        print(f"  {l:30} {n}")
    print()
    print("Split distribution (merged):")
    for sp, n in sorted(collections.Counter(c["split"] for c in all_cases).items()):
        print(f"  {sp:30} {n}")

    total_turns = sum(len(c["turns"]) for c in all_cases)
    print(f"\nTotal cases: {len(all_cases)}")
    print(f"Total turns: {total_turns}")
    print(f"Avg turns/case: {total_turns / len(all_cases):.1f}")

    # Sort deterministically by split / slice / case_id
    split_order = {"dev_smoke": 0, "dev_tune": 1, "test_core": 2, "test_stress": 3, "role_attribution": 4}
    all_cases.sort(key=lambda c: (split_order.get(c["split"], 99), c["slice"], c["case_id"]))

    MASTER_OUT.write_text(
        json.dumps(all_cases, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✓ Wrote {len(all_cases)} cases → {MASTER_OUT}")

    if all_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
