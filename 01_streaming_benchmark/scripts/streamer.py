#!/usr/bin/env python3
"""WebSocket streaming client for sending audio to STT server."""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import websockets

from noise import add_noise, NOISE_TYPES  # noqa: F401  (re-export for convenience)

SAMPLE_RATE = 16000


def read_wav_int16_mono_16k(path: Path) -> np.ndarray:
    with contextlib.closing(wave.open(str(path), "rb")) as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        src_rate = wav.getframerate()
        raw = wav.readframes(wav.getnframes())

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    elif sample_width == 1:
        data = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    elif sample_width == 4:
        data = (np.frombuffer(raw, dtype="<i4").astype(np.int32) >> 16).astype(np.int16)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if channels > 1:
        data = np.mean(data.reshape(-1, channels), axis=1).astype(np.int16)

    if src_rate != SAMPLE_RATE:
        duration = len(data) / float(src_rate)
        dst_len = max(1, int(round(duration * SAMPLE_RATE)))
        x_old = np.linspace(0.0, 1.0, num=len(data), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        data = np.clip(np.round(np.interp(x_new, x_old, data.astype(np.float32))), -32768, 32767).astype(np.int16)
    return data


async def _receiver_loop(
    ws: websockets.WebSocketClientProtocol,
    t0: float,
    events: list[dict[str, Any]],
    queue: asyncio.Queue[dict[str, Any]],
    state: dict[str, Any],
) -> None:
    try:
        async for msg in ws:
            ts_ms = (time.perf_counter() - t0) * 1000.0
            try:
                payload = json.loads(msg)
            except Exception:
                payload = {"type": "non_json", "raw": msg}
            event = {"ts_ms": round(ts_ms, 3), **payload}
            events.append(event)
            await queue.put(event)
    except websockets.exceptions.ConnectionClosed as exc:
        if exc.code != 1000:
            state["disconnect_count"] = state.get("disconnect_count", 0) + 1
        event = {"ts_ms": round((time.perf_counter() - t0) * 1000.0, 3), "type": "ws_closed", "code": exc.code}
        events.append(event)
        await queue.put(event)
    except Exception as exc:
        state["disconnect_count"] = state.get("disconnect_count", 0) + 1
        state["receiver_error"] = str(exc)
    finally:
        await queue.put({"type": "_done"})


async def stream_audio(
    ws_url: str,
    audio_samples: np.ndarray,
    chunk_ms: int = 160,
    realtime_speed: float = 1.0,
    tail_silence_ms: int = 2400,
    idle_timeout_ms: int = 1200,
    max_post_wait_ms: int = 8000,
    noise_type: str = "clean",
    snr_db: float = 20.0,
    cutoff_at_sample: int | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Stream audio to server. Returns (events, hyp_text, error_str).

    noise_type       : "clean" | "white" | "pink" | "babble" | "office" | "codec" | "echo"
    snr_db           : signal-to-noise ratio in dB (additive types only)
    cutoff_at_sample : if set, stop sending audio at this sample index and skip
                       tail silence (simulates agent barge-in / interruption)
    """
    # Apply noise before streaming (clean audio files remain untouched on disk)
    if noise_type != "clean":
        audio_samples = add_noise(audio_samples, noise_type, snr_db)

    # Cutoff: truncate audio at the given sample position
    if cutoff_at_sample is not None and 0 < cutoff_at_sample < len(audio_samples):
        audio_samples = audio_samples[:cutoff_at_sample]

    chunk_samples = max(1, int(round(SAMPLE_RATE * (chunk_ms / 1000.0))))
    chunk_sleep = (chunk_ms / 1000.0) / max(realtime_speed, 0.001)

    events: list[dict[str, Any]] = []
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    state: dict[str, Any] = {"disconnect_count": 0}

    t0 = time.perf_counter()

    async with websockets.connect(ws_url, max_size=10 * 1024 * 1024, open_timeout=20) as ws:
        recv_task = asyncio.create_task(_receiver_loop(ws, t0, events, queue, state))
        try:
            await ws.send(json.dumps({"command": "ping"}))
            await ws.send(json.dumps({"command": "reset"}))

            # Send audio chunks
            idx = 0
            while idx < len(audio_samples):
                chunk = audio_samples[idx:idx + chunk_samples]
                await ws.send(chunk.astype("<i2", copy=False).tobytes())
                idx += chunk_samples
                await asyncio.sleep(chunk_sleep)

            # Send tail silence (skipped in cutoff mode — no clean ending expected)
            if tail_silence_ms > 0 and cutoff_at_sample is None:
                tail_samples = int(round(SAMPLE_RATE * (tail_silence_ms / 1000.0)))
                zero = np.zeros(chunk_samples, dtype=np.int16)
                for _ in range(int(math.ceil(tail_samples / chunk_samples))):
                    await ws.send(zero.tobytes())
                    await asyncio.sleep(chunk_sleep)

            # Wait for final results
            last_event_at = time.monotonic()
            deadline = time.monotonic() + (max_post_wait_ms / 1000.0)
            idle_s = idle_timeout_ms / 1000.0

            while time.monotonic() < deadline:
                timeout = min(idle_s, max(0.05, deadline - time.monotonic()))
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    if time.monotonic() - last_event_at >= idle_s:
                        break
                    continue
                if event.get("type") == "_done":
                    break
                if event.get("type") in {"transcript", "error", "ws_closed"}:
                    last_event_at = time.monotonic()
        finally:
            await ws.close()
            recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await recv_task

    # Extract hypothesis text
    finals = [e for e in events if e.get("type") == "transcript" and e.get("is_final")]
    if finals:
        hyp_text = "".join(str(e.get("text", "")) for e in finals).strip()
    else:
        transcripts = [e for e in events if e.get("type") == "transcript"]
        hyp_text = str(transcripts[-1].get("text", "")).strip() if transcripts else ""

    return events, hyp_text, state.get("receiver_error", "")
