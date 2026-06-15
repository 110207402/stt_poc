#!/usr/bin/env python3
"""CER, latency, and RTF metric calculations."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from opencc import OpenCC

    _converter = OpenCC("s2tw")
except ImportError:
    _converter = None


def _s2tw(text: str) -> str:
    if _converter is not None:
        return _converter.convert(text)
    return text


def normalize_for_cer(text: str) -> str:
    converted = _s2tw(text or "")
    return "".join(converted.split())


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def cer_percent(hyp: str, ref: str) -> float | None:
    ref_norm = normalize_for_cer(ref)
    if not ref_norm:
        return None
    hyp_norm = normalize_for_cer(hyp)
    dist = levenshtein_distance(ref_norm, hyp_norm)
    return (dist / len(ref_norm)) * 100.0


def _avg(vals: list[float]) -> float | None:
    return float(sum(vals) / len(vals)) if vals else None


def _percentile(vals: list[float], q: float) -> float | None:
    return float(np.percentile(np.array(vals, dtype=np.float64), q)) if vals else None


def cutoff_recovery_rate(hyp: str, pre_cutoff_ref: str) -> float | None:
    """
    CER of the transcribed text against the pre-cutoff reference portion.
    100% = perfect recovery of what was said before the interruption.
    Returned as (1 - CER/100) * 100 so higher = better.
    """
    if not pre_cutoff_ref:
        return None
    c = cer_percent(hyp, pre_cutoff_ref)
    if c is None:
        return None
    return max(0.0, 100.0 - c)


def compute_run_metrics(
    events: list[dict[str, Any]],
    hyp_text: str,
    ref_text: str,
    audio_duration_s: float,
    disconnect_count: int = 0,
    pre_cutoff_ref: str = "",
) -> dict[str, Any]:
    transcripts = [e for e in events if e.get("type") == "transcript"]
    partials = [e for e in transcripts if not e.get("is_final")]
    finals = [e for e in transcripts if e.get("is_final")]
    errors = [e for e in events if e.get("type") == "error"]

    ttfp_ms = partials[0]["ts_ms"] if partials else None
    ttff_ms = finals[0]["ts_ms"] if finals else None
    last_ts = finals[-1]["ts_ms"] if finals else (transcripts[-1]["ts_ms"] if transcripts else None)
    e2e_ms = float(last_ts) if last_ts is not None else None
    rtf = (e2e_ms / (audio_duration_s * 1000.0)) if (e2e_ms and audio_duration_s > 0) else None

    latency_values = [float(e["latency_ms"]) for e in transcripts if isinstance(e.get("latency_ms"), (int, float))]

    return {
        "audio_duration_s": audio_duration_s,
        "ttfp_ms": ttfp_ms,
        "ttff_ms": ttff_ms,
        "e2e_ms": e2e_ms,
        "rtf": rtf,
        "avg_latency_ms": _avg(latency_values),
        "p50_latency_ms": _percentile(latency_values, 50),
        "p90_latency_ms": _percentile(latency_values, 90),
        "p95_latency_ms": _percentile(latency_values, 95),
        "min_latency_ms": min(latency_values) if latency_values else None,
        "partial_events": len(partials),
        "final_segments": len(finals),
        "error_count": len(errors),
        "disconnect_count": disconnect_count,
        "hyp_text": hyp_text,
        "ref_text": ref_text,
        "cer": cer_percent(hyp_text, ref_text),
        "pre_cutoff_ref": pre_cutoff_ref,
        "cutoff_recovery_rate": cutoff_recovery_rate(hyp_text, pre_cutoff_ref) if pre_cutoff_ref else None,
    }


def summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Group rows by (model, noise_type) and compute aggregate metrics.
    Sorted by (CER → E2E) so best performers appear first.
    """
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        noise = str(row.get("noise_type") or "clean")
        key = f"{row['model']}|||{noise}"
        by_key.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group_rows in by_key.items():
        model, noise_type = key.split("|||", 1)
        ok_rows = [r for r in group_rows if r.get("status") == "ok"]
        snr_vals = [float(r["snr_db"]) for r in group_rows if isinstance(r.get("snr_db"), (int, float))]

        s: dict[str, Any] = {
            "model": model,
            "noise_type": noise_type,
            "snr_db": snr_vals[0] if snr_vals else None,
            "runs_total": len(group_rows),
            "runs_ok": len(ok_rows),
            "success_rate": (len(ok_rows) / len(group_rows) * 100.0) if group_rows else 0.0,
        }
        for metric in ["cer", "ttfp_ms", "ttff_ms", "e2e_ms", "rtf", "avg_latency_ms", "p95_latency_ms", "cutoff_recovery_rate"]:
            vals = [float(r[metric]) for r in ok_rows if isinstance(r.get(metric), (int, float))]
            s[f"{metric}_mean"] = _avg(vals)
            s[f"{metric}_p50"] = _percentile(vals, 50)
        summary.append(s)

    summary.sort(key=lambda x: (
        x.get("cer_mean") if isinstance(x.get("cer_mean"), (int, float)) else float("inf"),
        x.get("e2e_ms_mean") if isinstance(x.get("e2e_ms_mean"), (int, float)) else float("inf"),
    ))
    return summary
