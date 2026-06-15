#!/usr/bin/env python3
"""
Phase 2 batch evaluator for offline ASR models:
  - fun-asr-nano     (sherpa-onnx OfflineFunASRNanoModelConfig)
  - fire-red-asr2    (sherpa-onnx OfflineFireRedAsrModelConfig)
  - qwen3-asr        (qwen-asr Python library, MPS)

Usage:
  python eval_batch.py \
    --models fun-asr-nano,fire-red-asr2,qwen3-asr \
    --cases-csv data/cases_seed.csv \
    --hokkien-csv data/hokkien_cases.csv \
    --mandarin-n 20 \
    --noise-types clean,codec,echo,babble \
    --snr 15 \
    --out-dir reports/phase2_batch/

Output CSV columns (unified with run_metrics.csv):
  model, noise_type, snr_db, case_id, category, repeat, status,
  audio_duration_s, cer, hyp_text, ref_text,
  inference_time_ms, rtf,
  ttfp_ms, ttff_ms, e2e_ms   <- NaN for batch models
"""

from __future__ import annotations

import argparse
import csv
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

SERVER_DIR = Path(__file__).parent.parent / "server"
BASE_DIR   = Path(__file__).parent

# ---------------------------------------------------------------------------
# CER helpers
# ---------------------------------------------------------------------------
import opencc
_converter = opencc.OpenCC("s2tw")

def simp_to_trad(text: str) -> str:
    return _converter.convert(text)

def cer(hyp: str, ref: str) -> float:
    """Character Error Rate (edit distance / len(ref))."""
    if not ref:
        return 0.0
    hyp = hyp.replace(" ", "")
    ref = ref.replace(" ", "")
    m, n = len(hyp), len(ref)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if hyp[i-1] == ref[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n] / n

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav_float32(path: str) -> tuple[np.ndarray, int]:
    """Load WAV as float32 in [-1, 1]."""
    with wave.open(path, "rb") as wf:
        sr   = wf.getframerate()
        n    = wf.getnframes()
        raw  = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return data, sr

# ---------------------------------------------------------------------------
# Noise injection (reuse noise.py)
# ---------------------------------------------------------------------------

import sys
sys.path.insert(0, str(BASE_DIR))
from noise import add_noise

# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------

@dataclass
class Case:
    case_id: str
    category: str
    reference_text: str
    audio_path: str

def load_mandarin_cases(csv_path: str, n: int, audio_dir: str) -> list[Case]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["case_id"]
            ap  = Path(audio_dir) / f"{cid}.wav"
            if not ap.exists():
                continue
            cases.append(Case(
                case_id        = cid,
                category       = row.get("category", ""),
                reference_text = row.get("reference_text", row.get("text", "")),
                audio_path     = str(ap),
            ))
            if len(cases) >= n:
                break
    return cases

def load_hokkien_cases(csv_path: str, audio_dir: str) -> list[Case]:
    cases = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["case_id"]
            ap  = Path(audio_dir) / f"{cid}.wav"
            if not ap.exists():
                continue
            cases.append(Case(
                case_id        = cid,
                category       = row.get("category", ""),
                reference_text = row["reference_text"],
                audio_path     = str(ap),
            ))
    return cases

# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

class FunASRNanoEngine:
    MODEL_DIR = SERVER_DIR / "sherpa-onnx-funasr-nano-int8-2025-12-30"
    # KV cache limit: 512 tokens ≈ 15s audio max per chunk
    MAX_CHUNK_S = 12

    def __init__(self):
        import sherpa_onnx
        d = self.MODEL_DIR
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_funasr_nano(
            encoder_adaptor = str(d / "encoder_adaptor.int8.onnx"),
            llm             = str(d / "llm.int8.onnx"),
            embedding       = str(d / "embedding.int8.onnx"),
            tokenizer       = str(d / "Qwen3-0.6B"),
            num_threads     = 4,
            itn             = True,
            debug           = False,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        """Split long audio into ≤12s chunks to fit KV cache, then concat."""
        chunk_size = int(self.MAX_CHUNK_S * sr)
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        parts = []
        for chunk in chunks:
            if len(chunk) < sr * 0.5:   # skip very short trailing silence
                continue
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sr, chunk)
            self.recognizer.decode_stream(stream)
            text = stream.result.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts)


class FireRedASR2Engine:
    MODEL_DIR = SERVER_DIR / "sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25"

    def __init__(self):
        import sherpa_onnx
        d = self.MODEL_DIR
        # from_fire_red_asr_ctc needs: model (single onnx), tokens
        model_files = list(d.glob("*.int8.onnx")) or list(d.glob("*.onnx"))
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_fire_red_asr_ctc(
            model       = str(model_files[0]),
            tokens      = str(d / "tokens.txt"),
            num_threads = 4,
            debug       = False,
        )

    def transcribe(self, audio: np.ndarray, sr: int) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()


class Qwen3ASREngine:
    def __init__(self):
        import torch
        from qwen_asr import Qwen3ASRModel
        self._model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-0.6B",
            dtype          = torch.float16,
            device_map     = "mps",
            max_inference_batch_size = 1,
            max_new_tokens = 512,
        )

    def transcribe(self, audio_path: str) -> str:
        results = self._model.transcribe(audio=audio_path, language=None)
        raw = results[0].text.strip()
        return simp_to_trad(raw)


ENGINE_REGISTRY = {
    "fun-asr-nano":  FunASRNanoEngine,
    "fire-red-asr2": FireRedASR2Engine,
    "qwen3-asr":     Qwen3ASREngine,
}

# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "noise_type", "snr_db", "case_id", "category", "repeat", "status",
    "audio_duration_s", "cer", "hyp_text", "ref_text",
    "inference_time_ms", "rtf",
    "ttfp_ms", "ttff_ms", "e2e_ms",
]

def run_evaluation(
    model_name:   str,
    engine,
    cases:        list[Case],
    noise_types:  list[str],
    snr_db:       float,
    writer:       csv.DictWriter,
    outfile,
    is_qwen:      bool = False,
    skip_set:     set  = None,
):
    total = len(cases) * len(noise_types)
    done  = 0
    for case in cases:
        audio_float, sr = load_wav_float32(case.audio_path)
        audio_int16 = (audio_float * 32767).astype(np.int16)
        dur_s = len(audio_float) / sr

        for noise_type in noise_types:
            done += 1
            if skip_set and (model_name, case.case_id, noise_type) in skip_set:
                print(f"  [{done:3d}/{total}] {case.case_id} {noise_type:6s} SKIP (already done)", flush=True)
                continue
            noisy_int16 = add_noise(audio_int16, noise_type, snr_db=snr_db)
            noisy = noisy_int16.astype(np.float32) / 32768.0

            row = {
                "model":          model_name,
                "noise_type":     noise_type,
                "snr_db":         "" if noise_type == "clean" else snr_db,
                "case_id":        case.case_id,
                "category":       case.category,
                "repeat":         1,
                "audio_duration_s": round(dur_s, 3),
                "ref_text":       case.reference_text,
                "ttfp_ms": "", "ttff_ms": "", "e2e_ms": "",
            }

            try:
                t0 = time.perf_counter()

                if is_qwen:
                    # Qwen3-ASR needs a file path — write noisy audio to temp WAV
                    import tempfile, wave as wv
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tmp_path = tf.name
                    with wv.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(sr)
                        wf.writeframes(noisy_int16.tobytes())
                    hyp = engine.transcribe(tmp_path)
                    import os; os.unlink(tmp_path)
                else:
                    hyp = engine.transcribe(noisy, sr)
                    # sherpa-onnx outputs simplified Chinese for some models
                    hyp = simp_to_trad(hyp)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                cer_val    = cer(hyp, case.reference_text)

                row.update({
                    "status":           "ok",
                    "cer":              round(cer_val * 100, 2),
                    "hyp_text":         hyp,
                    "inference_time_ms": round(elapsed_ms, 1),
                    "rtf":              round(elapsed_ms / 1000 / dur_s, 3),
                })

            except Exception as e:
                row.update({
                    "status":   "error",
                    "cer":      "",
                    "hyp_text": "",
                    "inference_time_ms": "",
                    "rtf":      "",
                    "error_msg": str(e)[:120],
                })

            writer.writerow(row)
            outfile.flush()
            cer_str = f"CER={row['cer']}%" if row["status"] == "ok" else "ERROR"
            rtf_str = f"RTF={row.get('rtf', '')}" if row["status"] == "ok" else ""
            print(f"  [{done:3d}/{total}] {case.case_id} {noise_type:6s} {cer_str} {rtf_str}", flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models",       default="fun-asr-nano,fire-red-asr2,qwen3-asr")
    ap.add_argument("--cases-csv",    default=str(BASE_DIR / "data/cases_seed.csv"))
    ap.add_argument("--hokkien-csv",  default=str(BASE_DIR / "data/hokkien_cases.csv"))
    ap.add_argument("--mandarin-n",   type=int, default=20)
    ap.add_argument("--audio-dir",    default=str(BASE_DIR / "data/audio"))
    ap.add_argument("--hokkien-dir",  default=str(BASE_DIR / "data/audio_hokkien"))
    ap.add_argument("--noise-types",  default="clean,codec,echo,babble")
    ap.add_argument("--snr",          type=float, default=15.0)
    ap.add_argument("--out-dir",      default=str(BASE_DIR / "reports/phase2_batch"))
    ap.add_argument("--append",       action="store_true", help="Append to existing CSV (no header)")
    ap.add_argument("--resume",       action="store_true", help="Resume: skip already-completed rows in existing CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "batch_metrics.csv"

    models      = [m.strip() for m in args.models.split(",")]
    noise_types = [n.strip() for n in args.noise_types.split(",")]

    mandarin_cases = load_mandarin_cases(args.cases_csv, args.mandarin_n, args.audio_dir) if args.mandarin_n > 0 else []
    hokkien_cases  = load_hokkien_cases(args.hokkien_csv, args.hokkien_dir)
    all_cases = mandarin_cases + hokkien_cases

    print(f"Cases: {len(mandarin_cases)} Mandarin + {len(hokkien_cases)} Hokkien = {len(all_cases)} total")
    print(f"Models: {models}")
    print(f"Noise: {noise_types} @ SNR={args.snr}dB")
    print(f"Total runs: {len(all_cases)} × {len(noise_types)} × {len(models)} = {len(all_cases)*len(noise_types)*len(models)}")
    print()

    # Build skip set from existing CSV if resuming
    skip_set = set()
    if args.resume and out_csv.exists():
        with open(out_csv, newline="", encoding="utf-8") as rf:
            for row in csv.DictReader(rf):
                if row.get("status") == "ok":
                    skip_set.add((row["model"], row["case_id"], row["noise_type"]))
        print(f"Resume mode: {len(skip_set)} completed rows found, will skip.")

    file_mode = "a" if (args.append or args.resume) else "w"
    with open(out_csv, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES + ["error_msg"], extrasaction="ignore")
        if file_mode == "w":
            writer.writeheader()

        for model_name in models:
            if model_name not in ENGINE_REGISTRY:
                print(f"Unknown model: {model_name}, skipping")
                continue

            print(f"\n[{model_name}] Loading engine...")
            t_load = time.perf_counter()
            engine = ENGINE_REGISTRY[model_name]()
            print(f"  Loaded in {time.perf_counter()-t_load:.1f}s")

            is_qwen = model_name == "qwen3-asr"
            run_evaluation(model_name, engine, all_cases, noise_types, args.snr, writer, f, is_qwen, skip_set)
            f.flush()
            print(f"[{model_name}] Done.")

    print(f"\nResults: {out_csv}")

if __name__ == "__main__":
    main()
