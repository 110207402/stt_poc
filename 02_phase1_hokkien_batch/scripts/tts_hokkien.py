#!/usr/bin/env python3
"""
Hokkien (Min Nan / 閩南語) TTS audio generation using facebook/mms-tts-nan.

Input:  benchmark/data/hokkien_cases.json
        Each case has a "script" list of segments with "poj" (Peh-ōe-jī romanisation).
Output: benchmark/data/audio_hokkien/{case_id}.wav  (16 kHz mono int16 PCM)

Usage:
    pip install transformers accelerate scipy
    python tts_hokkien.py
    python tts_hokkien.py --output-dir /custom/path
    python tts_hokkien.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
CASES_JSON = DATA_DIR / "hokkien_cases.json"
AUDIO_DIR  = DATA_DIR / "audio_hokkien"
MODEL_ID   = "facebook/mms-tts-nan"
SAMPLE_RATE = 16000


def load_model():
    try:
        from transformers import VitsModel, AutoTokenizer
    except ImportError:
        raise SystemExit(
            "Missing: pip install transformers accelerate scipy\n"
            "Model will be downloaded from HuggingFace (~145 MB) on first run."
        )
    import torch
    print(f"Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = VitsModel.from_pretrained(MODEL_ID)
    model.eval()
    sr = model.config.sampling_rate
    print(f"  Model loaded (sample_rate={sr})")
    return model, tokenizer, sr, torch


def synthesize_poj(poj_text: str, model, tokenizer, torch, sr: int) -> np.ndarray:
    """
    Synthesise a POJ-romanised sentence → int16 PCM at the model's native sample rate.
    Resamples to SAMPLE_RATE (16 kHz) if the model outputs a different rate.
    """
    inputs = tokenizer(poj_text, return_tensors="pt")
    with torch.no_grad():
        wav = model(**inputs).waveform[0].numpy()     # float32, range ≈ [-1, 1]

    # Resample if needed
    if sr != SAMPLE_RATE:
        n_src = len(wav)
        n_dst = max(1, int(round(n_src * SAMPLE_RATE / sr)))
        x_old = np.linspace(0.0, 1.0, num=n_src, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_dst, endpoint=False)
        wav = np.interp(x_new, x_old, wav).astype(np.float32)

    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16)


def build_case_audio(
    case: dict,
    model,
    tokenizer,
    torch,
    sr: int,
) -> np.ndarray:
    """
    Assemble one case's full audio: CLIENT speech segments interleaved
    with AGENT_SILENCE pauses.
    """
    segments: list[np.ndarray] = []

    for seg in case["script"]:
        if seg["speaker"] == "AGENT_SILENCE":
            silence_len = int(round(seg.get("pause_after", 2.0) * SAMPLE_RATE))
            segments.append(np.zeros(silence_len, dtype=np.int16))
            continue

        # CLIENT speech
        poj = seg.get("poj", "").strip()
        if not poj:
            continue

        pcm = synthesize_poj(poj, model, tokenizer, torch, sr)
        segments.append(pcm)

        pause = seg.get("pause_after", 0.0)
        if pause > 0:
            silence_len = int(round(pause * SAMPLE_RATE))
            segments.append(np.zeros(silence_len, dtype=np.int16))

    if not segments:
        return np.zeros(SAMPLE_RATE, dtype=np.int16)   # 1s silent fallback
    return np.concatenate(segments)


def write_wav(path: Path, pcm: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Hokkien TTS audio from hokkien_cases.json")
    parser.add_argument("--cases-json", default=str(CASES_JSON))
    parser.add_argument("--output-dir", default=str(AUDIO_DIR))
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-generate even if WAV already exists")
    args = parser.parse_args()

    cases_path = Path(args.cases_json)
    out_dir    = Path(args.output_dir)

    if not cases_path.exists():
        raise SystemExit(f"Cases file not found: {cases_path}")

    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(cases)} cases from {cases_path.name}")

    model, tokenizer, sr, torch = load_model()

    generated = skipped = failed = 0
    for case in cases:
        case_id = case["case_id"]
        out_path = out_dir / f"{case_id}.wav"

        if out_path.exists() and not args.overwrite:
            print(f"  SKIP {case_id} (already exists)")
            skipped += 1
            continue

        try:
            print(f"  GEN  {case_id} ...", end=" ", flush=True)
            pcm = build_case_audio(case, model, tokenizer, torch, sr)
            write_wav(out_path, pcm)
            duration = len(pcm) / SAMPLE_RATE
            print(f"{duration:.1f}s → {out_path}")
            generated += 1
        except Exception as exc:
            print(f"FAILED: {exc}")
            failed += 1

    print(f"\nDone: {generated} generated, {skipped} skipped, {failed} failed")
    print(f"Audio files: {out_dir}/")


if __name__ == "__main__":
    main()
