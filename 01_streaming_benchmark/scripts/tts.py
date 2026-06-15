#!/usr/bin/env python3
"""Edge-TTS based audio generation for benchmark cases.

Dialogue-gap mode (default):
  Splits each reference text at terminal punctuation (。！？!?) and inserts
  random silence between sentences, simulating the customer pausing while the
  agent responds.  A 2-channel call recording would have both sides; here we
  only have the client side, so gaps represent agent turns.
"""

from __future__ import annotations

import asyncio
import csv
import io
import random
import re
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import edge_tts
except ImportError:
    raise SystemExit("Missing: pip install edge-tts")

try:
    import soundfile as sf
except ImportError:
    raise SystemExit("Missing: pip install soundfile")

TARGET_SAMPLE_RATE = 16000
VOICES = [
    "zh-TW-HsiaoChenNeural",  # female
    "zh-TW-YunJheNeural",     # male
]

# Terminal punctuation that marks a sentence boundary
_SENTENCE_END_RE = re.compile(r'(?<=[。！？!?])\s*')

# Default silence gap range (seconds) — simulates agent response time
DEFAULT_GAP_MIN_S: float = 1.5
DEFAULT_GAP_MAX_S: float = 3.5


def split_sentences(text: str) -> list[str]:
    """Split text at Chinese / English terminal punctuation boundaries."""
    parts = _SENTENCE_END_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class TTSCase:
    case_id: str
    category: str
    reference_text: str
    audio_path: Path


def load_cases(csv_path: Path, audio_dir: Path) -> list[TTSCase]:
    if not csv_path.exists():
        raise FileNotFoundError(f"cases_seed.csv not found: {csv_path}")

    cases: list[TTSCase] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = (row.get("case_id") or "").strip()
            category = (row.get("category") or "").strip()
            ref = (row.get("reference_text") or "").strip()
            if not case_id or not ref:
                continue
            cases.append(TTSCase(
                case_id=case_id,
                category=category,
                reference_text=ref,
                audio_path=audio_dir / f"{case_id}.wav",
            ))
    if not cases:
        raise ValueError("No valid cases in CSV")
    return cases


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples
    if len(samples) == 0:
        return samples
    duration = len(samples) / float(src_rate)
    dst_len = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    interp = np.interp(x_new, x_old, samples.astype(np.float64))
    return interp


def _write_wav_16k_mono(path: Path, samples_f64: np.ndarray) -> None:
    pcm = np.clip(samples_f64 * 32767.0, -32768, 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


async def _tts_to_array(text: str, voice: str) -> np.ndarray:
    """Synthesize text → float64 numpy array at TARGET_SAMPLE_RATE."""
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]
    if not audio_bytes:
        raise RuntimeError(f"Edge-TTS returned empty audio for: {text[:40]!r}")
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float64")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return _resample_linear(data, sr, TARGET_SAMPLE_RATE)


async def _synthesize_one(
    case: TTSCase,
    voice: str,
    dialogue_gaps: bool = True,
    gap_min_s: float = DEFAULT_GAP_MIN_S,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
) -> None:
    """
    Generate TTS audio for a single case.

    When dialogue_gaps=True and the text contains multiple sentences, inserts
    random silence gaps between them (simulating agent turn silence).
    """
    sentences = split_sentences(case.reference_text) if dialogue_gaps else []

    if len(sentences) > 1:
        # Multi-sentence: generate each separately and interleave silence
        segments: list[np.ndarray] = []
        for i, sent in enumerate(sentences):
            seg = await _tts_to_array(sent, voice)
            segments.append(seg)
            if i < len(sentences) - 1:
                gap_s = random.uniform(gap_min_s, gap_max_s)
                silence = np.zeros(int(round(gap_s * TARGET_SAMPLE_RATE)), dtype=np.float64)
                segments.append(silence)
        combined = np.concatenate(segments)
    else:
        # Single sentence (or gaps disabled): generate full text in one call
        combined = await _tts_to_array(case.reference_text, voice)

    _write_wav_16k_mono(case.audio_path, combined)


async def generate_all(
    cases: list[TTSCase],
    overwrite: bool = False,
    max_concurrent: int = 5,
    dialogue_gaps: bool = True,
    gap_min_s: float = DEFAULT_GAP_MIN_S,
    gap_max_s: float = DEFAULT_GAP_MAX_S,
) -> tuple[int, int, int]:
    """
    Generate TTS audio for all cases. Returns (ok, skipped, failed).

    dialogue_gaps: insert silence between sentences to simulate agent turns
    gap_min_s / gap_max_s: random silence duration range (seconds)
    """
    ok = 0
    skipped = 0
    failed = 0
    sem = asyncio.Semaphore(max_concurrent)

    async def _do_one(idx: int, case: TTSCase) -> None:
        nonlocal ok, skipped, failed

        if case.audio_path.exists() and not overwrite:
            print(f"  [{idx}/{len(cases)}] {case.case_id} → skip (exists)")
            skipped += 1
            return

        voice = VOICES[idx % len(VOICES)]
        async with sem:
            try:
                await _synthesize_one(
                    case, voice,
                    dialogue_gaps=dialogue_gaps,
                    gap_min_s=gap_min_s,
                    gap_max_s=gap_max_s,
                )
                sents = split_sentences(case.reference_text) if dialogue_gaps else [case.reference_text]
                gap_note = f" {len(sents)}-turn" if len(sents) > 1 else ""
                print(f"  [{idx}/{len(cases)}] {case.case_id}{gap_note} → {case.audio_path.name} ({voice}) ✓")
                ok += 1
            except Exception as exc:
                print(f"  [{idx}/{len(cases)}] {case.case_id} → FAIL: {exc}")
                failed += 1

    tasks = [_do_one(i + 1, c) for i, c in enumerate(cases)]
    await asyncio.gather(*tasks)
    return ok, skipped, failed
