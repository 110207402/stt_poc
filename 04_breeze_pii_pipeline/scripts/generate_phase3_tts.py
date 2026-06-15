"""
Phase 3 TTS generation for KGI Breeze-ASR-26 PoC.

Reads benchmark/breeze_asr26/data/phase3_queries.csv (90 queries) and
synthesises one wav per query via Yating TTS:

  - language=mandarin   → m_seg with a Mandarin voice
  - language=hokkien    → h_seg with a Taiwanese (Hokkien) voice
  - language=codeswitch → m_seg with Mandarin + h_seg with Hokkien,
                          concatenated into a single wav (same speaker
                          gender across the join).

All audio is LINEAR16 / 16 kHz / mono so we can concat with the wave module.
Output: benchmark/breeze_asr26/audio/{case_id}.wav  + manifest.json
Resumable: existing wav files are skipped.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import pathlib
import sys
import time
import wave
from dataclasses import dataclass

import requests

API_KEY = "768b22d585833fbfb1409769fb58490a5c771f90"
ENDPOINT = "https://tts.api.yating.tw/v2/speeches/short"

ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "phase3_queries.csv"
OUT_DIR = ROOT / "audio"
MANIFEST_PATH = OUT_DIR / "manifest.json"

# Yating voice IDs. Mandarin uses zh_en_* (general TW Mandarin), Hokkien uses tai_*.
MANDARIN_VOICES = ["zh_en_female_1", "zh_en_male_1"]
HOKKIEN_VOICES = ["tai_female_1", "tai_male_1"]

# Pair index → (gender index): 0=female, 1=male. Same gender across an m+h join.
GENDER_PAIRS = [
    (MANDARIN_VOICES[0], HOKKIEN_VOICES[0]),  # both female
    (MANDARIN_VOICES[1], HOKKIEN_VOICES[1]),  # both male
]

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN = 0.3  # polite rate limit


@dataclass
class Query:
    case_id: str
    subtype: str
    language: str  # mandarin | hokkien | codeswitch
    pii_types: str
    pii_values: str
    domain_terms: str
    text: str
    m_seg: str
    h_seg: str


def load_queries(csv_path: pathlib.Path) -> list[Query]:
    rows: list[Query] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Query(
                    case_id=r["case_id"].strip(),
                    subtype=r["subtype"].strip(),
                    language=r["language"].strip(),
                    pii_types=r["pii_types"],
                    pii_values=r["pii_values"],
                    domain_terms=r["domain_terms"],
                    text=r["text"],
                    m_seg=r.get("m_seg", "") or "",
                    h_seg=r.get("h_seg", "") or "",
                )
            )
    return rows


def tts_generate(text: str, voice: str) -> bytes:
    """Call Yating TTS, return raw wav bytes (LINEAR16, 16 kHz, mono)."""
    headers = {"key": API_KEY, "Content-Type": "application/json"}
    body = {
        "input": {"text": text, "type": "text"},
        "voice": {"model": voice, "speed": 1.0, "pitch": 1.0, "energy": 1.0},
        "audioConfig": {"encoding": "LINEAR16", "sampleRate": "16K"},
    }
    resp = requests.post(ENDPOINT, headers=headers, json=body, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return base64.b64decode(data["audioContent"])


def concat_wavs(wav_bytes_list: list[bytes]) -> bytes:
    """Concatenate same-format LINEAR16 mono wavs into a single wav blob."""
    assert wav_bytes_list, "concat_wavs called with empty list"
    params = None
    frames = bytearray()
    for blob in wav_bytes_list:
        with wave.open(io.BytesIO(blob), "rb") as w:
            p = w.getparams()
            if params is None:
                params = p
            else:
                if (p.nchannels, p.sampwidth, p.framerate) != (
                    params.nchannels,
                    params.sampwidth,
                    params.framerate,
                ):
                    raise RuntimeError(
                        f"wav format mismatch: {p} vs {params} — cannot concat"
                    )
            frames.extend(w.readframes(p.nframes))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setparams(params)
        w.writeframes(bytes(frames))
    return out.getvalue()


def wav_duration_seconds(blob: bytes) -> float:
    with wave.open(io.BytesIO(blob), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def synthesise(query: Query, gender_idx: int) -> tuple[bytes, dict]:
    """Return (wav_bytes, voice_info_dict) for a single query."""
    m_voice, h_voice = GENDER_PAIRS[gender_idx]
    info: dict = {"language": query.language}

    if query.language == "mandarin":
        text = query.m_seg or query.text
        blob = tts_generate(text, m_voice)
        info["voice"] = m_voice
        info["segments"] = [{"voice": m_voice, "text": text}]
        return blob, info

    if query.language == "hokkien":
        text = query.h_seg or query.text
        blob = tts_generate(text, h_voice)
        info["voice"] = h_voice
        info["segments"] = [{"voice": h_voice, "text": text}]
        return blob, info

    if query.language == "codeswitch":
        if not query.m_seg or not query.h_seg:
            raise ValueError(
                f"{query.case_id}: codeswitch query missing m_seg or h_seg"
            )
        m_blob = tts_generate(query.m_seg, m_voice)
        time.sleep(SLEEP_BETWEEN)
        h_blob = tts_generate(query.h_seg, h_voice)
        joined = concat_wavs([m_blob, h_blob])
        info["voice"] = f"{m_voice}+{h_voice}"
        info["segments"] = [
            {"voice": m_voice, "text": query.m_seg},
            {"voice": h_voice, "text": query.h_seg},
        ]
        return joined, info

    raise ValueError(f"{query.case_id}: unknown language {query.language!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Only process first N queries (0 = all).",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated case_ids to process (overrides --limit).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-synthesise even if wav already exists.",
    )
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: queries CSV not found: {CSV_PATH}", file=sys.stderr)
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = load_queries(CSV_PATH)

    if args.only:
        wanted = {x.strip() for x in args.only.split(",") if x.strip()}
        queries = [q for q in queries if q.case_id in wanted]
    elif args.limit > 0:
        queries = queries[: args.limit]

    counts = {"mandarin": 0, "hokkien": 0, "codeswitch": 0}
    for q in queries:
        counts[q.language] = counts.get(q.language, 0) + 1
    print(
        f"Loaded {len(queries)} queries → "
        f"mandarin={counts.get('mandarin', 0)} "
        f"hokkien={counts.get('hokkien', 0)} "
        f"codeswitch={counts.get('codeswitch', 0)}"
    )
    print(f"Output dir: {OUT_DIR}\n")

    manifest: list[dict] = []
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            manifest = []
    by_id = {m["case_id"]: m for m in manifest}

    ok, skipped, failed = 0, 0, 0
    for i, q in enumerate(queries, 1):
        out_path = OUT_DIR / f"{q.case_id}.wav"
        # Same-gender alternation by index so we get a roughly 50/50 mix.
        gender_idx = i % 2

        if out_path.exists() and not args.force:
            print(f"  [{i:03d}/{len(queries)}] {q.case_id} ({q.language}) → SKIP (exists)")
            skipped += 1
            continue

        try:
            blob, info = synthesise(q, gender_idx)
            out_path.write_bytes(blob)
            dur = wav_duration_seconds(blob)
            size_kb = len(blob) / 1024
            print(
                f"  [{i:03d}/{len(queries)}] {q.case_id} ({q.language:10s}) "
                f"{info['voice']:32s} {dur:5.2f}s {size_kb:6.1f} KB"
            )
            by_id[q.case_id] = {
                "case_id": q.case_id,
                "subtype": q.subtype,
                "language": q.language,
                "voice": info["voice"],
                "segments": info["segments"],
                "duration_sec": round(dur, 3),
                "size_bytes": len(blob),
                "wav_path": f"audio/{q.case_id}.wav",
            }
            ok += 1
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            print(f"  [{i:03d}/{len(queries)}] {q.case_id} FAIL: {e}")
            failed += 1

    manifest = [by_id[k] for k in sorted(by_id.keys())]
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"\nDone. ok={ok} skipped={skipped} failed={failed}  "
        f"manifest={MANIFEST_PATH.relative_to(ROOT.parent)}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
