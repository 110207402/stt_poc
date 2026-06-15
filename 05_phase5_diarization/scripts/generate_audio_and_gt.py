"""
Phase 5 v2 — Local audio + ground truth generator.

Reads dialog specs (JSON), calls Azure Speech + Yating TTS per turn,
runs VAD on each turn, mixes to master timeline, writes three GT artifacts:

  audio/turns_raw/{case_id}/turn_NN_{speaker}.wav    — per-turn raw TTS
  audio/dialogs_clean/{case_id}.wav                   — mixed master audio
  data/ground_truth_per_case/speech/{case_id}.rttm   — VAD-active intervals (standard DER)
  data/ground_truth_per_case/turn/{case_id}.rttm     — turn ownership (role attribution)
  data/manifest.csv                                   — per-case metadata (durations, VAD ratio)

Resume-safe: skips cases with existing output.

Usage:
    cd benchmark/phase5_v2
    python scripts/generate_audio_and_gt.py
    python scripts/generate_audio_and_gt.py --only v2_smk001,v2_smk004
    python scripts/generate_audio_and_gt.py --spec-file data/dialog_specs/dev_smoke_12.json
"""

from __future__ import annotations
import argparse
import base64
import csv
import io
import json
import math
import os
import pathlib
import sys
import time
import wave
from dataclasses import dataclass

import numpy as np
import requests
from dotenv import load_dotenv

# ==========================================================================
# Config
# ==========================================================================
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
load_dotenv(PROJECT_ROOT / ".env")

V2_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_SPEC = V2_ROOT / "data" / "dialog_specs" / "dev_smoke_12.json"

AUDIO_DIR        = V2_ROOT / "audio"
AUDIO_TURNS_DIR  = AUDIO_DIR / "turns_raw"
AUDIO_CLEAN_DIR  = AUDIO_DIR / "dialogs_clean"
GT_DIR           = V2_ROOT / "data" / "ground_truth_per_case"
GT_SPEECH_DIR    = GT_DIR / "speech"
GT_TURN_DIR      = GT_DIR / "turn"
MANIFEST_CSV     = V2_ROOT / "data" / "manifest.csv"

SAMPLE_RATE = 16000
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_CALLS = 0.2

# Yating TTS API key (same as Phase 3, hokkien + codeswitch use this)
YATING_API_KEY = "768b22d585833fbfb1409769fb58490a5c771f90"
YATING_ENDPOINT = "https://tts.api.yating.tw/v2/speeches/short"

# Azure Speech (Mandarin) — from .env
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "").rstrip("/")

# ==========================================================================
# TTS providers
# ==========================================================================

def azure_tts(text: str, voice_id: str, pitch: str | None = None) -> bytes:
    """Azure Speech REST → raw wav bytes (LINEAR16, 16kHz, mono).

    If `pitch` is provided (e.g. '-1st', '+1st'), wraps the text in SSML
    <prosody pitch=...> tag to create acoustic variant of the same voice.
    Used by same_gender_similar M+M (one speaker pitch-shifted) so we can
    do same-gender stress with only 1 zh-TW Azure male voice.
    """
    if not AZURE_API_KEY or not AZURE_ENDPOINT:
        raise RuntimeError(
            f"Azure not configured. Set AZURE_API_KEY + AZURE_ENDPOINT in {PROJECT_ROOT}/.env"
        )
    url = f"{AZURE_ENDPOINT}/tts/cognitiveservices/v1"
    text_esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    inner = f"<prosody pitch='{pitch}'>{text_esc}</prosody>" if pitch else text_esc
    ssml = (
        f"<speak version='1.0' xml:lang='zh-TW'>"
        f"<voice name='{voice_id}'>{inner}</voice>"
        f"</speak>"
    )
    r = requests.post(url, headers={
        "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
    }, data=ssml.encode("utf-8"), timeout=REQUEST_TIMEOUT)
    if r.status_code != 200:
        raise RuntimeError(f"Azure Speech {r.status_code}: {r.text[:200]}")
    return r.content


# Voice pool → pitch shift for variant pools (codex's trick)
POOL_PITCH = {
    "azure_zh_tw_male_variant":   "-1st",  # 1 semitone lower
    "azure_zh_tw_female_alt2":    "+1st",  # 1 semitone higher
    # alt1 keeps default; differs from base via voice_id (HsiaoYu vs HsiaoChen)
}


def yating_tts(text: str, voice_id: str) -> bytes:
    """Yating TTS → raw wav bytes (LINEAR16, 16kHz, mono)."""
    body = {
        "input": {"text": text, "type": "text"},
        "voice": {"model": voice_id, "speed": 1.0, "pitch": 1.0, "energy": 1.0},
        "audioConfig": {"encoding": "LINEAR16", "sampleRate": "16K"},
    }
    r = requests.post(
        YATING_ENDPOINT,
        headers={"key": YATING_API_KEY, "Content-Type": "application/json"},
        json=body, timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return base64.b64decode(r.json()["audioContent"])


def is_azure_voice(voice_id: str) -> bool:
    return voice_id.startswith("zh-TW-") or voice_id.endswith("Neural")


def tts_call(text: str, voice_id: str, voice_pool: str | None = None) -> bytes:
    """Route to Azure or Yating based on voice id prefix.

    If voice_pool is in POOL_PITCH (e.g. `azure_zh_tw_male_variant`),
    applies SSML pitch shift so same voice_id renders as acoustic variant.
    """
    if is_azure_voice(voice_id):
        pitch = POOL_PITCH.get(voice_pool) if voice_pool else None
        return azure_tts(text, voice_id, pitch=pitch)
    return yating_tts(text, voice_id)


# ==========================================================================
# Audio I/O
# ==========================================================================

def wav_bytes_to_samples(blob: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(blob), "rb") as w:
        return (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16),
                w.getframerate())


def samples_to_wav_bytes(samples: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.clip(samples, -32768, 32767).astype(np.int16).tobytes())
    return out.getvalue()


# ==========================================================================
# VAD (frame-RMS energy detection)
# ==========================================================================

def frame_rms(samples: np.ndarray, frame_ms: int = 20, hop_ms: int = 10,
              rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """Return (start_indices, rms_per_frame)."""
    frame_len = int(rate * frame_ms / 1000)
    hop = int(rate * hop_ms / 1000)
    if len(samples) < frame_len:
        return np.array([0]), np.array([0.0])
    starts = np.arange(0, len(samples) - frame_len + 1, hop)
    rms = np.empty(len(starts))
    x = samples.astype(np.float64)
    for i, s in enumerate(starts):
        rms[i] = math.sqrt(float(np.mean(x[s:s + frame_len] ** 2)))
    return starts, rms


def detect_speech_intervals(samples: np.ndarray, tags: list[str] | None = None,
                            rate: int = SAMPLE_RATE) -> list[tuple[float, float]]:
    """
    VAD via frame-RMS thresholding.

    Returns list of (start_sec, end_sec) for speech-active intervals.
    Tuned for TTS audio (very clean, low noise floor).
    """
    tags = tags or []
    frame_ms, hop_ms = 20, 10
    starts, rms = frame_rms(samples, frame_ms, hop_ms, rate)
    if len(rms) == 0:
        return [(0.0, len(samples) / rate)]

    noise = float(np.percentile(rms, 10))
    high = float(np.percentile(rms, 95))
    threshold = max(120.0, noise + 0.08 * max(0.0, high - noise))

    mask = rms > threshold
    frame_len = int(rate * frame_ms / 1000)

    # Mask → intervals
    intervals = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < len(mask) and mask[j]:
            j += 1
        s = starts[i] / rate
        e = min(len(samples), starts[j - 1] + frame_len) / rate
        intervals.append((s, e))
        i = j

    if not intervals:
        return [(0.0, len(samples) / rate)]

    # Merge gaps < 150ms (preserve natural intra-sentence pauses but not silence)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        if s - merged[-1][1] <= 0.15:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Drop islands < 80ms (or 50ms for backchannel turns)
    is_backchannel = any("backchannel" in t for t in tags)
    min_dur = 0.05 if is_backchannel else 0.08
    merged = [(s, e) for s, e in merged if e - s >= min_dur]

    if not merged:
        return [(0.0, len(samples) / rate)]

    # Pad each side by 60ms (natural breath)
    duration = len(samples) / rate
    padded = [(max(0.0, s - 0.06), min(duration, e + 0.06)) for s, e in merged]

    return padded


# ==========================================================================
# Dialog generation
# ==========================================================================

@dataclass
class TurnData:
    turn_idx: int
    speaker: str
    role: str
    samples: np.ndarray            # trimmed (leading/trailing silence removed)
    duration: float                # of trimmed
    local_intervals: list[tuple[float, float]]  # VAD intervals on trimmed timeline
    timing: dict
    start_time: float = 0.0        # set during scheduling


def generate_dialog(spec: dict, force: bool = False) -> dict:
    """Generate one dialog. Returns stats dict."""
    cid = spec["case_id"]
    out_wav      = AUDIO_CLEAN_DIR / f"{cid}.wav"
    speech_rttm  = GT_SPEECH_DIR / f"{cid}.rttm"
    turn_rttm    = GT_TURN_DIR / f"{cid}.rttm"

    if not force and out_wav.exists() and speech_rttm.exists() and turn_rttm.exists():
        # Already done — return cached stats
        with wave.open(str(out_wav), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        return {"case_id": cid, "cached": True, "duration": dur}

    print(f"  [{cid}] generating...")
    turn_dir = AUDIO_TURNS_DIR / cid
    turn_dir.mkdir(parents=True, exist_ok=True)

    # === Step 1: TTS each turn, run VAD ===
    turn_data: list[TurnData] = []
    for turn in spec["turns"]:
        idx = turn["turn_idx"]
        speaker = turn["speaker"]
        role = turn["role"]
        participant = spec["participants"][speaker]
        voice_id = participant["voice_id"]
        voice_pool = participant.get("voice_pool")
        raw_path = turn_dir / f"turn_{idx:02d}_{speaker}.wav"

        if raw_path.exists() and not force:
            samples, _ = wav_bytes_to_samples(raw_path.read_bytes())
        else:
            blob = tts_call(turn["text"], voice_id, voice_pool=voice_pool)
            raw_path.write_bytes(blob)
            samples, _ = wav_bytes_to_samples(blob)
            time.sleep(SLEEP_BETWEEN_CALLS)

        # VAD on raw turn audio
        raw_intervals = detect_speech_intervals(samples, tags=turn.get("tags", []))

        # Trim leading/trailing silence (mixer uses this trimmed audio)
        vad_start = raw_intervals[0][0]
        vad_end = raw_intervals[-1][1]
        trim_start_sample = int(vad_start * SAMPLE_RATE)
        trim_end_sample = int(vad_end * SAMPLE_RATE)
        trimmed = samples[trim_start_sample:trim_end_sample]
        # Re-localize VAD intervals to trimmed timeline
        local_intervals = [
            (max(0, s - vad_start), min(len(trimmed) / SAMPLE_RATE, e - vad_start))
            for s, e in raw_intervals
        ]

        turn_data.append(TurnData(
            turn_idx=idx,
            speaker=speaker,
            role=role,
            samples=trimmed,
            duration=len(trimmed) / SAMPLE_RATE,
            local_intervals=local_intervals,
            timing=turn["timing"],
        ))

    # === Step 2: Schedule master timeline ===
    # Two overlap modes:
    #   (a) start_relation == "same_time_as_prev" → simultaneous_start slice
    #       Both speakers begin near the SAME time. offset = lag of 2nd speaker
    #       (e.g. offset=-0.22 → 2nd speaker starts 0.22s after 1st).
    #       Schedule from prev.START_TIME (not prev_end).
    #   (b) start_relation absent → agent_barge_in / customer_barge_in
    #       2nd speaker cuts in near END of prev's turn.
    #       Schedule as prev_end + offset (negative → starts before prev ends).
    for i, td in enumerate(turn_data):
        if i == 0:
            td.start_time = 0.0
            continue
        prev = turn_data[i - 1]
        prev_end = prev.start_time + prev.duration
        prev_pause = prev.timing.get("forced_pause_after_sec", 0.0)
        prev_end_with_pause = prev_end + prev_pause

        if td.timing.get("overlap_with_prev"):
            offset = td.timing.get("overlap_offset_sec", -1.0)
            if td.timing.get("start_relation") == "same_time_as_prev":
                # simultaneous start: schedule from prev.start, offset = small lag
                td.start_time = max(0.0, prev.start_time + abs(offset))
            else:
                # barge_in: schedule from prev_end
                td.start_time = max(0.0, prev_end + offset)
        else:
            delay = td.timing.get("delay_before_sec", 0.4)
            # Earliest safe start = max end of ALL prior turns (handles simultaneous_start
            # case where prev (turn i-1) may end before turn i-2).
            max_prior_end = max(
                (other.start_time + other.duration for other in turn_data[:i]),
                default=prev_end
            )
            td.start_time = max(max_prior_end, prev_end_with_pause) + delay

    # === Step 3: Mix audio onto master ===
    total_dur = max(td.start_time + td.duration for td in turn_data)
    master = np.zeros(int(total_dur * SAMPLE_RATE) + 1, dtype=np.float32)
    for td in turn_data:
        s = int(td.start_time * SAMPLE_RATE)
        e = s + len(td.samples)
        gain = 0.7 if td.timing.get("overlap_with_prev") else 1.0
        bg_db = td.timing.get("background_gain_db")
        if td.role == "other" and bg_db is not None:
            gain *= 10 ** (bg_db / 20)
        master[s:e] += td.samples.astype(np.float32) * gain

    master_int16 = np.clip(master * 0.9, -32768, 32767).astype(np.int16)
    out_wav.write_bytes(samples_to_wav_bytes(master_int16))

    # === Step 4: Write GT RTTMs ===
    speech_lines = []
    turn_lines = []
    speech_total = 0.0
    for td in turn_data:
        # Turn RTTM: full trimmed turn duration
        turn_lines.append(
            f"SPEAKER {cid} 1 {td.start_time:.3f} {td.duration:.3f} "
            f"<NA> <NA> {td.role} <NA> <NA>"
        )
        # Speech RTTM: per-VAD-interval mapped to master timeline
        for ls, le in td.local_intervals:
            seg_start = td.start_time + ls
            seg_dur = le - ls
            if seg_dur > 0:
                speech_lines.append(
                    f"SPEAKER {cid} 1 {seg_start:.3f} {seg_dur:.3f} "
                    f"<NA> <NA> {td.role} <NA> <NA>"
                )
                speech_total += seg_dur

    speech_rttm.write_text("\n".join(speech_lines) + "\n", encoding="utf-8")
    turn_rttm.write_text("\n".join(turn_lines) + "\n", encoding="utf-8")

    return {
        "case_id": cid,
        "split": spec["split"],
        "slice": spec["slice"],
        "language": spec["language_profile"],
        "cached": False,
        "duration": round(total_dur, 3),
        "speech_duration": round(speech_total, 3),
        "vad_active_ratio": round(speech_total / total_dur, 4),
        "n_turns": len(turn_data),
        "n_speech_segments": len(speech_lines),
        "audio_path": str(out_wav.relative_to(V2_ROOT)),
        "speech_rttm": str(speech_rttm.relative_to(V2_ROOT)),
        "turn_rttm": str(turn_rttm.relative_to(V2_ROOT)),
    }


# ==========================================================================
# Main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-file", default=str(DEFAULT_SPEC))
    parser.add_argument("--only", default="", help="Comma-separated case_ids")
    parser.add_argument("--force", action="store_true", help="Re-generate even if cached")
    args = parser.parse_args()

    spec_path = pathlib.Path(args.spec_file)
    if not spec_path.exists():
        print(f"ERROR: spec file not found: {spec_path}", file=sys.stderr)
        return 2

    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        specs = [s for s in specs if s["case_id"] in wanted]

    # Set up output dirs
    for d in [AUDIO_TURNS_DIR, AUDIO_CLEAN_DIR, GT_SPEECH_DIR, GT_TURN_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(specs)} dialogs from {spec_path.name}")
    print(f"Azure: {bool(AZURE_API_KEY)} | Yating: {bool(YATING_API_KEY)}\n")

    stats = []
    for i, spec in enumerate(specs, 1):
        cid = spec["case_id"]
        print(f"[{i:02d}/{len(specs)}] {cid} ({spec['slice']}, {spec['language_profile']})")
        try:
            stat = generate_dialog(spec, force=args.force)
            if not stat["cached"]:
                print(f"  → dur={stat['duration']:.1f}s, VAD ratio={stat['vad_active_ratio']:.1%}, "
                      f"speech segs={stat['n_speech_segments']}")
            else:
                print(f"  → cached (dur={stat['duration']:.1f}s)")
            stats.append(stat)
        except Exception as e:
            print(f"  FAIL: {type(e).__name__}: {e}", file=sys.stderr)

    # Write manifest CSV
    real_stats = [s for s in stats if not s.get("cached")]
    if real_stats:
        fieldnames = list(real_stats[0].keys())
        with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(real_stats)
        print(f"\nWrote {MANIFEST_CSV.relative_to(V2_ROOT)}")

    # Sanity stats
    ratios = [s["vad_active_ratio"] for s in stats if "vad_active_ratio" in s and not s.get("cached")]
    if ratios:
        print(f"\n=== VAD active ratio sanity ===")
        print(f"  mean: {np.mean(ratios):.1%}")
        print(f"  min:  {min(ratios):.1%}")
        print(f"  max:  {max(ratios):.1%}")
        print(f"  Healthy range: 0.60-0.85.")
        if np.mean(ratios) < 0.55:
            print(f"  ⚠️  VAD may be too strict, missing speech")
        elif np.mean(ratios) > 0.90:
            print(f"  ⚠️  VAD may be too lax, keeping silence")
        else:
            print(f"  ✓ in healthy range")

    print(f"\nDone. {len(stats)} dialogs.")
    print(f"Audio: {AUDIO_CLEAN_DIR.relative_to(V2_ROOT)}/")
    print(f"GT speech: {GT_SPEECH_DIR.relative_to(V2_ROOT)}/")
    print(f"GT turn:   {GT_TURN_DIR.relative_to(V2_ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
