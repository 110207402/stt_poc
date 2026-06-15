"""
B5 baseline — Embedding-based role attribution (closed-set binary).

Task: For each role_attribution test case, predict which of the 2 speakers is agent
      using ONLY acoustic embeddings (no text, no LLM).

Method:
  1. Build global agent / customer prototypes from dev_smoke + dev_tune
     - Extract ECAPA-TDNN embedding for each agent turn → average → agent_proto
     - Same for customer turns → customer_proto
  2. For each test case, extract embedding per spec speaker
     (concat all their turn audio, run embedder once)
  3. Predict: speaker with higher cosine_sim to agent_proto - sim to customer_proto = agent

This is the proper "pyannote-style enrollment" test in synthetic data.
Expected: synthetic data uses same voice pool for both roles → embeddings overlap →
acoustic alone can't strongly discriminate. Result will tell us the ceiling.

If accuracy > 75% → real prod B-architecture viable
If accuracy < 70% → must use text (B2) or wait for real recordings
"""

from __future__ import annotations
import json
import csv
import pathlib
import collections
import wave
import contextlib

import numpy as np
import torch
from scipy.io import wavfile
from speechbrain.inference.speaker import EncoderClassifier

V2_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC_PATH = V2_ROOT / "data" / "dialog_specs" / "all_dialogs.json"
TURNS_DIR = V2_ROOT / "audio" / "turns_raw"
RESULTS_DIR = V2_ROOT / "results" / "role_attribution"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_wav_resampled(path: pathlib.Path, target_sr: int = 16000) -> torch.Tensor:
    """Load wav via scipy → (1, samples) tensor at target_sr.

    Our TTS audio is already 16kHz mono LINEAR16 so no resample needed.
    """
    sr, samples = wavfile.read(str(path))
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    elif samples.dtype == np.int32:
        samples = samples.astype(np.float32) / 2147483648.0
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    assert sr == target_sr, f"Expected {target_sr}Hz but got {sr}Hz in {path}"
    return torch.from_numpy(samples).float().unsqueeze(0)


def embed_turns(turn_paths: list[pathlib.Path], embedder) -> np.ndarray:
    """Average ECAPA embeddings across multiple turn audio files."""
    embeddings = []
    for p in turn_paths:
        if not p.exists(): continue
        wav = load_wav_resampled(p)
        # ECAPA expects (batch, samples)
        with torch.no_grad():
            emb = embedder.encode_batch(wav).squeeze().cpu().numpy()
        embeddings.append(emb)
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def build_global_prototypes(cases, embedder):
    """Mean agent and customer embeddings across dev_smoke + dev_tune."""
    agent_embs, customer_embs = [], []
    n_agent_turns = n_customer_turns = 0

    for c in cases:
        if c["split"] not in ("dev_smoke", "dev_tune"):
            continue
        case_dir = TURNS_DIR / c["case_id"]
        for t in c["turns"]:
            if t.get("speaker") not in ("agent", "customer"):
                continue
            wav_path = case_dir / f"turn_{t['turn_idx']:02d}_{t['speaker']}.wav"
            if not wav_path.exists():
                continue
            wav = load_wav_resampled(wav_path)
            if wav.shape[-1] < 8000:  # < 0.5s → skip short
                continue
            with torch.no_grad():
                emb = embedder.encode_batch(wav).squeeze().cpu().numpy()
            if t["speaker"] == "agent":
                agent_embs.append(emb); n_agent_turns += 1
            else:
                customer_embs.append(emb); n_customer_turns += 1

    if not agent_embs or not customer_embs:
        raise RuntimeError("No agent or customer enrollment turns found")

    agent_proto = np.mean(agent_embs, axis=0)
    customer_proto = np.mean(customer_embs, axis=0)
    print(f"  Built prototypes from {n_agent_turns} agent turns + {n_customer_turns} customer turns")
    print(f"  Prototype self-sim agent↔customer: {cosine_sim(agent_proto, customer_proto):.4f} (high = poorly separated)")
    return agent_proto, customer_proto


def predict_b5(case: dict, agent_proto, customer_proto, embedder) -> dict:
    """For each spec speaker, compute embedding then predict by cosine sim."""
    case_dir = TURNS_DIR / case["case_id"]
    speaker_embs = {}

    for spk in ("agent", "customer"):
        turn_paths = [
            case_dir / f"turn_{t['turn_idx']:02d}_{t['speaker']}.wav"
            for t in case["turns"] if t.get("speaker") == spk
        ]
        speaker_embs[spk] = embed_turns(turn_paths, embedder)

    if speaker_embs.get("agent") is None or speaker_embs.get("customer") is None:
        return {"predicted_agent_label": None, "confidence": 0.0, "error": "missing_embedding"}

    sims = {}
    for spk, emb in speaker_embs.items():
        sa = cosine_sim(emb, agent_proto)
        sc = cosine_sim(emb, customer_proto)
        sims[spk] = sa - sc

    pred = max(sims.keys(), key=lambda k: sims[k])
    conf = abs(sims["agent"] - sims["customer"])
    return {
        "predicted_agent_label": pred,
        "confidence": round(conf, 4),
        "agent_speaker_score": round(sims["agent"], 4),
        "customer_speaker_score": round(sims["customer"], 4),
    }


def main():
    print("Loading ECAPA-TDNN embedder (speechbrain/spkrec-ecapa-voxceleb)...")
    embedder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/spkrec_ecapa",
        run_opts={"device": "cpu"},
    )
    print("✓ Embedder ready (ECAPA-TDNN, 192-dim, trained on VoxCeleb)\n")

    all_cases = json.loads(SPEC_PATH.read_text())

    print("Building global prototypes from dev_smoke + dev_tune...")
    agent_proto, customer_proto = build_global_prototypes(all_cases, embedder)
    print()

    role_attr_cases = [c for c in all_cases if c["split"] == "role_attribution" and len(c["participants"]) == 2]
    print(f"Evaluating B5 on {len(role_attr_cases)} closed-set 2-speaker cases...\n")

    results = []
    for i, c in enumerate(role_attr_cases, 1):
        b5 = predict_b5(c, agent_proto, customer_proto, embedder)
        truth = "agent"
        correct = b5.get("predicted_agent_label") == truth

        row = {
            "case_id": c["case_id"],
            "slice": c["slice"],
            "signal_strength": c["role_signal_strength"],
            "recording_start": c["recording_start"],
            "agent_voice": c["participants"]["agent"]["voice_id"],
            "customer_voice": c["participants"]["customer"]["voice_id"],
            "same_voice": c["participants"]["agent"]["voice_id"] == c["participants"]["customer"]["voice_id"],
            "b5_pred": b5.get("predicted_agent_label"),
            "b5_correct": correct,
            "b5_conf": b5.get("confidence", 0),
            "agent_speaker_score": b5.get("agent_speaker_score", 0),
            "customer_speaker_score": b5.get("customer_speaker_score", 0),
        }
        results.append(row)

        if i % 5 == 0 or i == len(role_attr_cases):
            print(f"  [{i:2}/{len(role_attr_cases)}] {c['case_id']} ({c['role_signal_strength']:6}) — B5={correct} (conf={b5.get('confidence',0):.3f})")

    # Summary
    n = len(results)
    correct = sum(1 for r in results if r["b5_correct"])
    print(f"\n{'=' * 78}\nB5 (ECAPA-TDNN embedding) — overall accuracy\n{'=' * 78}\n")
    print(f"Overall: {correct}/{n} = {correct/n:.1%}\n")

    print("Stratified by signal_strength:")
    by_sig = collections.defaultdict(list)
    for r in results:
        by_sig[r["signal_strength"]].append(r)
    for sig in ["strong", "medium", "weak"]:
        rows = by_sig.get(sig, [])
        if not rows: continue
        c = sum(1 for r in rows if r["b5_correct"])
        print(f"  {sig:8} {c}/{len(rows)} = {c/len(rows):.1%}")

    print("\nStratified by same_voice (when agent and customer use SAME TTS voice):")
    by_sv = collections.defaultdict(list)
    for r in results:
        by_sv[r["same_voice"]].append(r)
    for sv in (False, True):
        rows = by_sv.get(sv, [])
        if not rows: continue
        c = sum(1 for r in rows if r["b5_correct"])
        label = "DIFFERENT voices" if not sv else "SAME voice (extreme stress)"
        print(f"  {label:32} {c}/{len(rows)} = {c/len(rows):.1%}")

    print("\nStratified by recording_start:")
    by_rec = collections.defaultdict(list)
    for r in results:
        by_rec[r["recording_start"]].append(r)
    for rec in ["full_call", "mid_call"]:
        rows = by_rec.get(rec, [])
        if not rows: continue
        c = sum(1 for r in rows if r["b5_correct"])
        print(f"  {rec:10} {c}/{len(rows)} = {c/len(rows):.1%}")

    out = RESULTS_DIR / "role_attribution_b5_embedder.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader(); w.writerows(results)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
