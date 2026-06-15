#!/usr/bin/env python3
"""Report generation: CSV detail, CSV summary, and Markdown comparison."""

from __future__ import annotations

import csv
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any


_RUN_FIELDS = [
    "model", "noise_type", "snr_db", "case_id", "category", "repeat", "status",
    "audio_duration_s", "cer", "hyp_text", "ref_text",
    "ttfp_ms", "ttff_ms", "e2e_ms", "rtf",
    "avg_latency_ms", "p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "min_latency_ms",
    "partial_events", "final_segments", "error_count", "disconnect_count",
    "error_msg",
]

_SUMMARY_FIELDS = [
    "model", "noise_type", "snr_db", "runs_total", "runs_ok", "success_rate",
    "cer_mean", "cer_p50",
    "ttfp_ms_mean", "ttfp_ms_p50",
    "ttff_ms_mean", "ttff_ms_p50",
    "e2e_ms_mean", "e2e_ms_p50",
    "rtf_mean", "rtf_p50",
    "avg_latency_ms_mean", "avg_latency_ms_p50",
    "p95_latency_ms_mean", "p95_latency_ms_p50",
]


def _fmt(val: Any, decimals: int = 2) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def write_run_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(summary: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            writer.writerow({k: _fmt(v) for k, v in row.items()})


def _noise_label(noise_type: str, snr_db: Any) -> str:
    if noise_type == "clean":
        return "clean"
    snr_str = f"{snr_db:.0f}dB" if isinstance(snr_db, (int, float)) else "?dB"
    return f"{noise_type}({snr_str})"


def write_markdown(summary: list[dict[str, Any]], run_rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_runs = len(run_rows)
    ok_runs = sum(1 for r in run_rows if r.get("status") == "ok")
    noise_conditions = sorted(set(str(r.get("noise_type") or "clean") for r in run_rows))

    lines: list[str] = []
    lines.append("# STT Model Comparison Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append(f"Total runs: {total_runs} | OK: {ok_runs} | Failed: {total_runs - ok_runs}")
    if len(noise_conditions) > 1 or noise_conditions[0] != "clean":
        lines.append(f"Noise conditions tested: {', '.join(_noise_label(c, next((r.get('snr_db') for r in run_rows if r.get('noise_type') == c), None)) for c in noise_conditions)}")
    lines.append("")

    # ── Overall ranking table ────────────────────────────────────────────────
    lines.append("## Overall Ranking (sorted by CER → E2E)")
    lines.append("")
    lines.append("| Rank | Model | Noise | CER (%) | TTFF (ms) | E2E (ms) | RTF | P95 Lat (ms) | Success |")
    lines.append("|------|-------|-------|---------|-----------|----------|-----|-------------|---------|")

    for i, s in enumerate(summary, 1):
        noise_lbl = _noise_label(str(s.get("noise_type") or "clean"), s.get("snr_db"))
        lines.append(
            f"| {i} "
            f"| {s['model']} "
            f"| {noise_lbl} "
            f"| {_fmt(s.get('cer_mean'))} "
            f"| {_fmt(s.get('ttff_ms_mean'), 0)} "
            f"| {_fmt(s.get('e2e_ms_mean'), 0)} "
            f"| {_fmt(s.get('rtf_mean'), 3)} "
            f"| {_fmt(s.get('p95_latency_ms_mean'), 0)} "
            f"| {_fmt(s.get('success_rate'), 1)}% |"
        )
    lines.append("")

    # ── Noise robustness table (if multiple noise conditions) ────────────────
    if len(noise_conditions) > 1:
        lines.append("## Noise Robustness (CER % by Model × Condition)")
        lines.append("")

        # Collect unique models in original order
        seen_models: list[str] = []
        for s in summary:
            m = str(s["model"])
            if m not in seen_models:
                seen_models.append(m)

        # Build lookup: {model: {noise_label: cer_mean}}
        cer_table: dict[str, dict[str, str]] = defaultdict(dict)
        for s in summary:
            m = str(s["model"])
            lbl = _noise_label(str(s.get("noise_type") or "clean"), s.get("snr_db"))
            cer_table[m][lbl] = _fmt(s.get("cer_mean"))

        col_labels = [
            _noise_label(c, next((r.get("snr_db") for r in run_rows if r.get("noise_type") == c), None))
            for c in noise_conditions
        ]
        header = "| Model | " + " | ".join(col_labels) + " | CER Δ (clean→worst) |"
        sep = "|-------|" + "|".join(["-------"] * len(col_labels)) + "|---------------------|"
        lines.append(header)
        lines.append(sep)

        for model in seen_models:
            row_vals = [cer_table[model].get(lbl, "") for lbl in col_labels]
            # Compute degradation: worst noisy CER − clean CER
            clean_lbl = _noise_label("clean", None)
            clean_cer_str = cer_table[model].get(clean_lbl, "")
            noisy_cers = [float(cer_table[model][lbl]) for lbl in col_labels
                          if lbl != clean_lbl and cer_table[model].get(lbl, "")]
            if clean_cer_str and noisy_cers:
                delta = max(noisy_cers) - float(clean_cer_str)
                delta_str = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
            else:
                delta_str = ""
            lines.append("| " + model + " | " + " | ".join(row_vals) + f" | {delta_str} |")

        lines.append("")

    # ── Per-model detail ─────────────────────────────────────────────────────
    lines.append("## Per-Model Details")
    lines.append("")

    for s in summary:
        noise_lbl = _noise_label(str(s.get("noise_type") or "clean"), s.get("snr_db"))
        lines.append(f"### {s['model']} — {noise_lbl}")
        lines.append("")
        lines.append(f"- Runs: {s['runs_ok']}/{s['runs_total']} OK")
        lines.append(f"- CER mean: {_fmt(s.get('cer_mean'))}% | p50: {_fmt(s.get('cer_p50'))}%")
        lines.append(f"- TTFP mean: {_fmt(s.get('ttfp_ms_mean'), 0)} ms")
        lines.append(f"- TTFF mean: {_fmt(s.get('ttff_ms_mean'), 0)} ms")
        lines.append(f"- E2E mean: {_fmt(s.get('e2e_ms_mean'), 0)} ms | p50: {_fmt(s.get('e2e_ms_p50'), 0)} ms")
        lines.append(f"- RTF mean: {_fmt(s.get('rtf_mean'), 3)} | p50: {_fmt(s.get('rtf_p50'), 3)}")
        lines.append(f"- Avg latency mean: {_fmt(s.get('avg_latency_ms_mean'), 0)} ms")
        lines.append(f"- P95 latency mean: {_fmt(s.get('p95_latency_ms_mean'), 0)} ms")
        lines.append("")

    # ── Worst CER cases ──────────────────────────────────────────────────────
    ok_rows = [r for r in run_rows if r.get("status") == "ok" and isinstance(r.get("cer"), (int, float))]
    if ok_rows:
        worst = sorted(ok_rows, key=lambda r: r["cer"], reverse=True)[:10]
        lines.append("## Worst CER Cases (top 10)")
        lines.append("")
        lines.append("| Model | Noise | Case | CER (%) | Ref | Hyp |")
        lines.append("|-------|-------|------|---------|-----|-----|")
        for r in worst:
            noise_lbl = _noise_label(str(r.get("noise_type") or "clean"), r.get("snr_db"))
            ref = (r.get("ref_text") or "")[:30]
            hyp = (r.get("hyp_text") or "")[:30]
            lines.append(f"| {r['model']} | {noise_lbl} | {r['case_id']} | {_fmt(r['cer'])} | {ref} | {hyp} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
