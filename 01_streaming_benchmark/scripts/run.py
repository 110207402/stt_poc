#!/usr/bin/env python3
"""
STT Benchmark Runner — single entry point.

Usage examples:
    # Full run with dialogue gaps + clean audio (default)
    python run.py

    # Quick smoke test
    python run.py --max-cases 3 --models paraformer,zipformer-zh-sm --repeats 1

    # Test noise robustness (clean + white 20dB + pink 10dB)
    python run.py --noise-types clean,white,pink --noise-snr 20

    # Only generate TTS audio (skip benchmark)
    python run.py --skip-benchmark

    # Benchmark only (TTS audio already exists)
    python run.py --skip-tts
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import sys
import time
from pathlib import Path
from typing import Any

from tts import load_cases, generate_all
from streamer import read_wav_int16_mono_16k, stream_audio, SAMPLE_RATE
from metrics import compute_run_metrics, summarize_by_model
from noise import save_noisy_wav, NOISE_TYPES as ALL_NOISE_TYPES
from bargein import cutoff_audio, overlap_audio, select_bargein_cases
from server import start_server, stop_server
from report import write_run_csv, write_summary_csv, write_markdown

DEFAULT_MODELS = ["paraformer", "paraformer-tri", "zipformer-zh-xl", "zipformer-zh-sm", "zipformer-zh-xl-t", "zipformer-zh-sm-t"]
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
CASES_CSV = DATA_DIR / "cases_seed.csv"
SERVER_SCRIPT = BASE_DIR.parent / "server" / "stt_server_pipeline.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="STT streaming benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Models ────────────────────────────────────────────────────────────────
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated model names")

    # ── Test dimensions ───────────────────────────────────────────────────────
    p.add_argument("--repeats", type=int, default=3,
                    help="Repeat count per case per noise condition")
    p.add_argument("--max-cases", type=int, default=0,
                    help="Limit number of cases (0 = all 60)")

    # ── Noise ─────────────────────────────────────────────────────────────────
    p.add_argument("--noise-types", default="clean",
                    help=f"Comma-separated noise types: {','.join(ALL_NOISE_TYPES)}")
    p.add_argument("--noise-snr", type=float, default=20.0,
                    help="SNR in dB for noisy conditions (single value; overridden by --noise-snr-levels)")
    p.add_argument("--noise-snr-levels", default="",
                    help="Comma-separated SNR levels in dB, e.g. 10,15,20 (runs each noise type at all levels)")
    p.add_argument("--generate-noise-wavs", action="store_true",
                    help="Pre-save noisy WAV files to data/audio_noisy/ for auditing, then exit")

    # ── Dialogue gaps (TTS) ───────────────────────────────────────────────────
    p.add_argument("--no-dialogue-gaps", action="store_true",
                    help="Disable silence gaps between sentences in TTS audio")
    p.add_argument("--gap-min-s", type=float, default=1.5,
                    help="Min silence gap duration (seconds) between sentences")
    p.add_argument("--gap-max-s", type=float, default=3.5,
                    help="Max silence gap duration (seconds) between sentences")

    # ── Streaming ─────────────────────────────────────────────────────────────
    p.add_argument("--chunk-ms", type=int, default=160,
                    help="Audio chunk size in ms")
    p.add_argument("--realtime-speed", type=float, default=1.0,
                    help="Streaming speed multiplier (1.0 = real-time)")
    p.add_argument("--tail-silence-ms", type=int, default=2400,
                    help="Silence appended after audio to flush STT endpoint detection")

    # ── Server ────────────────────────────────────────────────────────────────
    p.add_argument("--base-port", type=int, default=18000,
                    help="Starting port (each model gets base-port + index)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--startup-timeout", type=float, default=120,
                    help="Server startup timeout in seconds")
    p.add_argument("--python", default=sys.executable,
                    help="Python interpreter for server subprocess")
    p.add_argument("--server-script", default=str(SERVER_SCRIPT),
                    help="Path to stt_server_pipeline.py")

    # ── Domain dictionary ─────────────────────────────────────────────────────
    _default_hw = str(BASE_DIR / "data" / "kgi_hotwords.txt")
    _default_cr = str(BASE_DIR / "data" / "kgi_corrections.json")
    p.add_argument("--hotwords-file", default=_default_hw,
                    help="Paraformer hot-word biasing file (auto-detected if exists)")
    p.add_argument("--hotwords-score", type=float, default=1.5,
                    help="Hot-word boost score, 1.0–2.0 (default: 1.5)")
    p.add_argument("--corrections-file", default=_default_cr,
                    help="Post-processing corrections JSON (all models)")
    p.add_argument("--no-domain-dict", action="store_true",
                    help="Disable both hotwords and corrections")

    # ── Barge-in testing ──────────────────────────────────────────────────────
    p.add_argument("--test-mode", default="standard",
                    choices=["standard", "cutoff", "overlap", "all"],
                    help="standard = normal benchmark; cutoff/overlap = barge-in modes; all = standard+cutoff+overlap")
    p.add_argument("--bargein-cases", type=int, default=10,
                    help="Number of cases for barge-in tests (randomly sampled, stratified by duration)")
    p.add_argument("--cutoff-ratio", type=float, default=0.0,
                    help="Cutoff position (0.0 = random 50-80%%; e.g. 0.6 = 60%%)")
    p.add_argument("--overlap-ratio", type=float, default=0.5,
                    help="Position where agent speech starts overlapping (0.0-1.0)")
    p.add_argument("--overlap-agent-text", default="請稍等，我來幫您查詢",
                    help="Agent utterance text for overlap TTS synthesis")

    # ── Pipeline control ──────────────────────────────────────────────────────
    p.add_argument("--skip-tts", action="store_true",
                    help="Skip TTS generation (assume audio already exists)")
    p.add_argument("--skip-benchmark", action="store_true",
                    help="Only generate TTS audio, skip benchmark")
    p.add_argument("--overwrite-tts", action="store_true",
                    help="Re-generate TTS even if audio files already exist")

    return p.parse_args()


async def run_benchmark(
    cases: list[Any],
    models: list[str],
    noise_types: list[str],
    args: argparse.Namespace,
    report_dir: Path,
    hotwords_file: str = "",
    hotwords_file_simp: str = "",
    corrections_file: str = "",
    all_rows: list[dict[str, Any]] | None = None,
    test_mode: str = "standard",
) -> list[dict[str, Any]]:
    """Run all models × noise conditions, return flat list of result rows."""
    if all_rows is None:
        all_rows = []
    server_script = Path(args.server_script)

    if not server_script.exists():
        print(f"ERROR: Server script not found: {server_script}")
        sys.exit(1)

    for model_idx, model in enumerate(models):
        port = args.base_port + model_idx
        log_path = report_dir / "logs" / f"{model}.log"

        # -t transducer models use simplified Chinese hotwords for better vocab coverage
        hw_file = hotwords_file_simp if (model.endswith("-t") and hotwords_file_simp) else hotwords_file

        print(f"\n  [{model}] Starting server on port {port}...", end=" ", flush=True)
        if hw_file:
            print(f"(hotwords: {Path(hw_file).name})", end=" ", flush=True)
        try:
            proc, ws_url, log_f = await start_server(
                python_bin=args.python,
                server_script=server_script,
                model=model,
                host=args.host,
                port=port,
                startup_timeout=args.startup_timeout,
                log_path=log_path,
                hotwords_file=hw_file,
                hotwords_score=args.hotwords_score,
                corrections_file=corrections_file,
            )
            print("ready")
        except Exception as exc:
            print(f"FAILED: {exc}")
            for noise_type in noise_types:
                for case in cases:
                    for rep in range(1, args.repeats + 1):
                        all_rows.append({
                            "model": model,
                            "noise_type": noise_type,
                            "snr_db": args.noise_snr if noise_type != "clean" else None,
                            "case_id": case.case_id,
                            "category": case.category,
                            "repeat": rep,
                            "status": "server_fail",
                            "error_msg": str(exc),
                            "ref_text": case.reference_text,
                        })
            continue

        try:
            total = len(cases) * len(noise_types) * args.repeats
            done = 0

            for noise_type in noise_types:
                noise_label = (
                    "clean" if noise_type == "clean"
                    else f"{noise_type} {args.noise_snr:.0f}dB"
                )
                snr_val = args.noise_snr if noise_type != "clean" else None
                print(f"\n    Noise: {noise_label}")

                for case in cases:
                    if not case.audio_path.exists():
                        print(f"      SKIP {case.case_id}: audio not found")
                        for rep in range(1, args.repeats + 1):
                            all_rows.append({
                                "model": model,
                                "noise_type": noise_type,
                                "snr_db": snr_val,
                                "case_id": case.case_id,
                                "category": case.category,
                                "repeat": rep,
                                "status": "no_audio",
                                "ref_text": case.reference_text,
                            })
                            done += 1
                        continue

                    audio_raw = read_wav_int16_mono_16k(case.audio_path)
                    audio_dur = len(audio_raw) / float(SAMPLE_RATE)

                    for rep in range(1, args.repeats + 1):
                        done += 1
                        cutoff_at = None
                        pre_cutoff_ref = ""

                        # ── Barge-in: cutoff ─────────────────────────────
                        if test_mode == "cutoff":
                            audio, actual_ratio = cutoff_audio(audio_raw, args.cutoff_ratio)
                            cutoff_at = len(audio)
                            # Approximate pre-cutoff reference (proportional char slice)
                            ref_chars = case.reference_text
                            pre_cutoff_ref = ref_chars[:max(1, int(len(ref_chars) * actual_ratio))]
                        # ── Barge-in: overlap ────────────────────────────
                        elif test_mode == "overlap":
                            audio, _overlap_dur = await overlap_audio(
                                audio_raw,
                                args.overlap_agent_text,
                                overlap_start_ratio=args.overlap_ratio,
                            )
                        else:
                            audio = audio_raw

                        try:
                            events, hyp, err_str = await stream_audio(
                                ws_url=ws_url,
                                audio_samples=audio,
                                chunk_ms=args.chunk_ms,
                                realtime_speed=args.realtime_speed,
                                tail_silence_ms=args.tail_silence_ms,
                                noise_type=noise_type,
                                snr_db=args.noise_snr,
                                cutoff_at_sample=cutoff_at,
                            )
                            m = compute_run_metrics(
                                events, hyp, case.reference_text, audio_dur,
                                pre_cutoff_ref=pre_cutoff_ref,
                            )
                            row: dict[str, Any] = {
                                "model": model,
                                "noise_type": noise_type,
                                "snr_db": snr_val,
                                "case_id": case.case_id,
                                "category": case.category,
                                "repeat": rep,
                                "test_mode": test_mode,
                                "status": "ok" if not err_str else "error",
                                "error_msg": err_str,
                                **m,
                            }
                        except Exception as exc:
                            row = {
                                "model": model,
                                "noise_type": noise_type,
                                "snr_db": snr_val,
                                "case_id": case.case_id,
                                "category": case.category,
                                "repeat": rep,
                                "test_mode": test_mode,
                                "status": "error",
                                "error_msg": str(exc),
                                "ref_text": case.reference_text,
                            }

                        all_rows.append(row)
                        cer_str = f"CER={row.get('cer', '?'):.1f}%" if isinstance(row.get("cer"), (int, float)) else "CER=N/A"
                        e2e_str = f"E2E={row.get('e2e_ms', '?'):.0f}ms" if isinstance(row.get("e2e_ms"), (int, float)) else "E2E=N/A"
                        print(f"      [{done}/{total}] {case.case_id} r{rep} {cer_str} {e2e_str}")

        finally:
            await stop_server(proc)
            try:
                log_f.close()
            except Exception:
                pass

        ok_count = sum(
            1 for r in all_rows
            if r["model"] == model and r["status"] == "ok"
        )
        total_for_model = len(cases) * len(noise_types) * args.repeats
        print(f"  [{model}] Done: {ok_count}/{total_for_model} OK")

    return all_rows


async def main() -> None:
    args = parse_args()

    # Parse lists
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    noise_types = [t.strip() for t in args.noise_types.split(",") if t.strip()]

    if not models:
        print("ERROR: No models specified")
        sys.exit(1)
    if not noise_types:
        noise_types = ["clean"]

    # Expand SNR levels: if --noise-snr-levels given, create (noise_type, snr) pairs
    if args.noise_snr_levels:
        snr_levels = [float(s.strip()) for s in args.noise_snr_levels.split(",") if s.strip()]
    else:
        snr_levels = [args.noise_snr]

    dialogue_gaps = not args.no_dialogue_gaps

    # ── Load cases ─────────────────────────────────────────────────────────
    cases = load_cases(CASES_CSV, AUDIO_DIR)
    if args.max_cases > 0:
        cases = cases[:args.max_cases]

    # ── Domain dictionary ──────────────────────────────────────────────────
    hotwords_file = ""
    hotwords_file_simp = ""
    corrections_file = ""
    if not args.no_domain_dict:
        hw_path = Path(args.hotwords_file)
        cr_path = Path(args.corrections_file)
        hotwords_file = str(hw_path) if hw_path.exists() else ""
        corrections_file = str(cr_path) if cr_path.exists() else ""
        # Auto-detect simplified hotwords for -t transducer models
        simp_path = hw_path.with_name(hw_path.stem + "_simp" + hw_path.suffix)
        hotwords_file_simp = str(simp_path) if simp_path.exists() else ""

    print(f"Loaded {len(cases)} cases from {CASES_CSV.name}")
    print(f"Models: {', '.join(models)}")
    print(f"Noise conditions: {', '.join(noise_types)}" +
          (f" @ SNR {args.noise_snr:.0f}dB" if any(n != "clean" for n in noise_types) else ""))
    print(f"Repeats: {args.repeats} | Dialogue gaps: {'on' if dialogue_gaps else 'off'}")
    if hotwords_file:
        print(f"Hotwords (non-transducer): {Path(hotwords_file).name} (score={args.hotwords_score})")
    if hotwords_file_simp:
        print(f"Hotwords (-t transducer):  {Path(hotwords_file_simp).name} (score={args.hotwords_score}, 簡體版 69% in-vocab)")
    if corrections_file:
        print(f"Corrections (all models): {Path(corrections_file).name}")

    # ── Pre-generate noisy WAVs (optional) ────────────────────────────────
    if args.generate_noise_wavs:
        noisy_dir = DATA_DIR / "audio_noisy"
        print(f"\n[Pre-gen] Saving noisy WAVs → {noisy_dir}/")
        cases_for_noise = load_cases(CASES_CSV, AUDIO_DIR)
        if args.max_cases > 0:
            cases_for_noise = cases_for_noise[:args.max_cases]
        saved = 0
        for case in cases_for_noise:
            if not case.audio_path.exists():
                continue
            audio = read_wav_int16_mono_16k(case.audio_path)
            for nt in noise_types:
                if nt == "clean":
                    continue
                for snr in snr_levels:
                    suffix = f"{snr:.0f}dB" if nt not in ("codec", "echo") else ""
                    fname = f"{case.case_id}_{nt}{suffix}.wav"
                    out_path = noisy_dir / fname
                    if not out_path.exists() or args.overwrite_tts:
                        save_noisy_wav(audio, nt, snr, out_path)
                        saved += 1
                    if nt in ("codec", "echo"):
                        break  # no SNR sweep for transform types
        print(f"  Saved {saved} noisy WAV files to {noisy_dir}/")
        return

    # ── Phase 1: TTS ───────────────────────────────────────────────────────
    if not args.skip_tts:
        print(f"\n[Phase 1/3] TTS Audio Generation")
        if dialogue_gaps:
            print(f"  Dialogue gaps: {args.gap_min_s:.1f}–{args.gap_max_s:.1f}s silence between sentences")
        ok, skipped, failed = await generate_all(
            cases,
            overwrite=args.overwrite_tts,
            dialogue_gaps=dialogue_gaps,
            gap_min_s=args.gap_min_s,
            gap_max_s=args.gap_max_s,
        )
        print(f"  TTS done: {ok} generated, {skipped} skipped, {failed} failed")
        if failed > 0:
            print("  WARNING: Some TTS generations failed")
    else:
        print(f"\n[Phase 1/3] TTS skipped (--skip-tts)")

    # ── Phase 2: Benchmark ─────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = BASE_DIR / "reports" / stamp

    if not args.skip_benchmark:
        # Determine test modes to run
        test_mode_arg = args.test_mode
        if test_mode_arg == "all":
            test_modes = ["standard", "cutoff", "overlap"]
        else:
            test_modes = [test_mode_arg]

        # For barge-in modes, select a stratified subset of cases
        bargein_cases = select_bargein_cases(cases, n=args.bargein_cases)

        print(f"\n[Phase 2/3] STT Benchmark")
        print(f"  Test modes: {', '.join(test_modes)}")
        if len(snr_levels) > 1:
            print(f"  SNR levels: {snr_levels} dB")
        t_start = time.monotonic()
        all_rows: list[dict[str, Any]] = []
        try:
            for test_mode_cur in test_modes:
                cases_cur = bargein_cases if test_mode_cur in ("cutoff", "overlap") else cases
                if test_mode_cur != "standard":
                    print(f"\n  [Barge-in: {test_mode_cur}] {len(cases_cur)} cases × {args.repeats} repeats")
                for snr in snr_levels:
                    args_copy = argparse.Namespace(**vars(args))
                    args_copy.noise_snr = snr
                    await run_benchmark(
                        cases_cur, models, noise_types, args_copy, report_dir,
                        hotwords_file=hotwords_file,
                        hotwords_file_simp=hotwords_file_simp,
                        corrections_file=corrections_file,
                        all_rows=all_rows,
                        test_mode=test_mode_cur,
                    )
        except KeyboardInterrupt:
            elapsed = time.monotonic() - t_start
            print(f"\n  [中斷] 已收到 Ctrl+C，儲存目前已完成的資料 ({elapsed:.1f}s elapsed, {len(all_rows)} runs)")
            if not all_rows:
                print("  無已完成資料，結束。")
                return
        elapsed = time.monotonic() - t_start
        print(f"\n  Benchmark finished in {elapsed:.1f}s ({len(all_rows)} total runs)")
    else:
        print(f"\n[Phase 2/3] Benchmark skipped (--skip-benchmark)")
        all_rows = []

    # ── Phase 3: Report ────────────────────────────────────────────────────
    if all_rows:
        print(f"\n[Phase 3/3] Generating Reports → {report_dir}/")
        summary = summarize_by_model(all_rows)

        run_csv = report_dir / "run_metrics.csv"
        sum_csv = report_dir / "model_summary.csv"
        md_path = report_dir / "comparison.md"

        write_run_csv(all_rows, run_csv)
        write_summary_csv(summary, sum_csv)
        write_markdown(summary, all_rows, md_path)

        print(f"  -> {run_csv}")
        print(f"  -> {sum_csv}")
        print(f"  -> {md_path}")

        # Print quick summary to stdout
        print(f"\n  {'Model':<22} {'Noise':<12} {'CER%':>6} {'TTFF':>7} {'E2E':>7}")
        print(f"  {'-'*22} {'-'*12} {'-'*6} {'-'*7} {'-'*7}")
        for s in summary:
            noise_lbl = (
                "clean" if s.get("noise_type") == "clean"
                else f"{s.get('noise_type')}({s.get('snr_db', '?'):.0f}dB)"
            )
            print(
                f"  {s['model']:<22} {noise_lbl:<12} "
                f"{_fmt(s.get('cer_mean')):>6} "
                f"{_fmt(s.get('ttff_ms_mean'), 0):>7} "
                f"{_fmt(s.get('e2e_ms_mean'), 0):>7}"
            )
    else:
        print(f"\n[Phase 3/3] No results to report")

    print("\nDone!")


def _fmt(val: Any, decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


if __name__ == "__main__":
    asyncio.run(main())
