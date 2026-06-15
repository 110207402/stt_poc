#!/usr/bin/env python3
"""
Noise injection utilities for STT robustness testing.

Supported noise types:
  clean   — no noise (pass-through)
  white   — white noise (flat spectrum)
  pink    — 1/f pink noise (more natural ambient sound)
  babble  — overlapping speech noise (café / crowded-room)
  office  — HVAC hum + fan + keyboard clicks
  codec   — G.711 μ-law codec degradation (8 kHz telephone)
  echo    — phone echo: delayed attenuated reflections

SNR (Signal-to-Noise Ratio) is calculated relative to the RMS power of
the speech signal, so silent sections do not inflate noise amplitude.
codec and echo are signal transformations rather than additive noise;
the snr_db parameter is ignored for those two types.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

NOISE_TYPES = ["clean", "white", "pink", "babble", "office", "codec", "echo"]
SAMPLE_RATE = 16000


# ── Basic noise generators ────────────────────────────────────────────────────

def _white_noise(n: int) -> np.ndarray:
    """Unit-std white noise, float32."""
    return np.random.randn(n).astype(np.float32)


def _pink_noise(n: int) -> np.ndarray:
    """
    Approximate 1/f pink noise via FFT spectral shaping.
    Shapes white noise so that power ∝ 1/f, giving a warmer,
    more natural-sounding ambient noise than white noise.
    Returns unit-std float32 array.
    """
    white = np.random.randn(n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    spectrum /= np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n).astype(np.float32)
    std = float(np.std(pink))
    if std > 1e-10:
        pink /= std
    return pink


# ── Realistic phone noise generators ─────────────────────────────────────────

def _babble_noise(n: int, num_speakers: int = 4) -> np.ndarray:
    """
    Simulate babble noise (café / crowded room) by summing N band-pass
    filtered noise streams, each with random formant-like peaks to give
    a speech-like spectral envelope.  Speech band: 300–3 400 Hz.
    Returns unit-std float32 array.
    """
    result = np.zeros(n, dtype=np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)
    for _ in range(num_speakers):
        white = np.random.randn(n).astype(np.float64)
        spectrum = np.fft.rfft(white)
        # Hard band-pass: 300–3400 Hz
        mask = (freqs >= 300) & (freqs <= 3400)
        spectrum *= mask
        # Add random formant emphasis
        f0 = float(np.random.uniform(80, 250))
        harmonics = np.arange(f0, 3400, f0)
        for hf in harmonics:
            idx = int(np.argmin(np.abs(freqs - hf)))
            half_w = max(1, n // 4000)
            lo, hi = max(0, idx - half_w), min(len(spectrum), idx + half_w + 1)
            spectrum[lo:hi] *= np.random.uniform(1.5, 3.5)
        stream = np.fft.irfft(spectrum, n).astype(np.float32)
        std = float(np.std(stream))
        result += stream / max(std, 1e-10)
    result = result.astype(np.float32)
    std = float(np.std(result))
    if std > 1e-10:
        result /= std
    return result


def _office_noise(n: int) -> np.ndarray:
    """
    Simulate office background noise:
      - HVAC / air-conditioning: low-frequency brownian rumble (< 300 Hz)
      - 50 Hz mains hum + harmonics
      - Keyboard clicks: sparse random impulses (~2/s)
    Returns unit-std float32 array.
    """
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)

    # Brownian (red) noise: power ∝ 1/f², low-pass at 300 Hz
    white = np.random.randn(n).astype(np.float64)
    spec = np.fft.rfft(white)
    f_safe = freqs.copy()
    f_safe[0] = 1.0
    spec /= f_safe ** 2
    spec[freqs > 300] = 0
    hvac = np.fft.irfft(spec, n).astype(np.float32)
    hvac_std = float(np.std(hvac))
    if hvac_std > 1e-10:
        hvac /= hvac_std

    # 50 Hz hum + harmonics
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    hum = (
        np.sin(2 * np.pi * 50 * t) * 0.30
        + np.sin(2 * np.pi * 100 * t) * 0.15
        + np.sin(2 * np.pi * 150 * t) * 0.07
    )

    # Sparse keyboard clicks (~2 per second)
    clicks = np.zeros(n, dtype=np.float32)
    num_clicks = max(1, int(n / SAMPLE_RATE * 2))
    positions = np.random.randint(0, n, size=num_clicks)
    lengths = np.random.randint(3, 10, size=num_clicks)
    for pos, clen in zip(positions, lengths):
        amp = float(np.random.uniform(0.4, 1.0)) * float(np.random.choice([-1, 1]))
        clicks[pos : min(n, pos + clen)] = amp

    # Mix: HVAC 50%, hum 30%, clicks 20%
    noise = 0.50 * hvac + 0.30 * hum + 0.20 * clicks
    std = float(np.std(noise))
    if std > 1e-10:
        noise /= std
    return noise


def _codec_distort(signal_f32: np.ndarray) -> np.ndarray:
    """
    Simulate G.711 μ-law codec degradation (telephone bandwidth):
      1. Low-pass filter to 3 400 Hz (telephone band-limit)
      2. Downsample 16 kHz → 8 kHz
      3. μ-law encode → quantise to 8-bit → decode
      4. Upsample 8 kHz → 16 kHz
    Returns float32 array, same length as input, values in [-1, 1].
    """
    n = len(signal_f32)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)

    # 1. Telephone band-limit (300–3400 Hz hard mask)
    spec = np.fft.rfft(signal_f32.astype(np.float64))
    mask = (freqs >= 300) & (freqs <= 3400)
    spec *= mask
    bandlim = np.fft.irfft(spec, n).astype(np.float32)

    # 2. Downsample by 2 (16 kHz → 8 kHz)
    down = bandlim[::2]

    # 3. μ-law encode / decode
    mu = 255.0
    x = np.clip(down, -1.0, 1.0).astype(np.float64)
    encoded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    quantized = np.round(encoded * 127.5) / 127.5            # 8-bit quantisation
    decoded = np.sign(quantized) * (np.expm1(np.abs(quantized) * np.log1p(mu)) / mu)
    decoded = decoded.astype(np.float32)

    # 4. Upsample 8 kHz → 16 kHz via linear interpolation
    x_old = np.linspace(0.0, 1.0, num=len(decoded), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n, endpoint=False)
    upsampled = np.interp(x_new, x_old, decoded).astype(np.float32)
    return upsampled


def _echo_distort(signal_f32: np.ndarray) -> np.ndarray:
    """
    Simulate phone handset / room echo:
      - 1st reflection at ~150 ms, −8 dB
      - 2nd reflection at ~300 ms, −14 dB
    Returns float32 array, same length as input.
    """
    result = signal_f32.copy().astype(np.float64)
    reflections = [(150, -8), (300, -14)]  # (ms, dB)
    for delay_ms, attn_db in reflections:
        delay = int(round(SAMPLE_RATE * delay_ms / 1000.0))
        gain = 10.0 ** (attn_db / 20.0)
        result[delay:] += gain * signal_f32[: len(signal_f32) - delay].astype(np.float64)
    # Normalise to prevent clipping while keeping relative dynamics
    peak = float(np.max(np.abs(result)))
    if peak > 1.0:
        result /= peak
    return result.astype(np.float32)


# ── SNR helper ────────────────────────────────────────────────────────────────

def _snr_amplitude(signal_f32: np.ndarray, snr_db: float) -> float:
    """
    Compute noise amplitude (std) needed to achieve target SNR.

    SNR_dB = 10 * log10(P_signal / P_noise)
    => P_noise = P_signal / 10^(SNR_dB/10)
    => amp_noise = sqrt(P_noise)
    """
    signal_power = float(np.mean(signal_f32.astype(np.float64) ** 2))
    if signal_power < 1e-10:
        return 0.0  # silent signal — don't add noise
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(noise_power))


# ── Public API ────────────────────────────────────────────────────────────────

def add_noise(audio_int16: np.ndarray, noise_type: str, snr_db: float = 20.0) -> np.ndarray:
    """
    Apply noise / codec distortion to PCM int16 audio.

    For additive types (white, pink, babble, office):
        noise is scaled to the given SNR relative to speech RMS.

    For transform types (codec, echo):
        snr_db is ignored; the signal is distorted in-place.

    Parameters
    ----------
    audio_int16 : np.ndarray  (dtype int16, 16 kHz mono)
    noise_type  : one of NOISE_TYPES
    snr_db      : target SNR in dB (additive types only)

    Returns
    -------
    np.ndarray  (dtype int16) — distorted audio, clipped to [-32768, 32767]
    """
    if noise_type == "clean":
        return audio_int16

    if noise_type not in NOISE_TYPES:
        raise ValueError(f"Unknown noise_type {noise_type!r}. Choose from: {NOISE_TYPES}")

    signal_f = audio_int16.astype(np.float32) / 32768.0
    n = len(signal_f)

    # ── Transform types ───────────────────────────────────────────────────
    if noise_type == "codec":
        distorted = _codec_distort(signal_f)
        return (np.clip(distorted, -1.0, 1.0) * 32768.0).astype(np.int16)

    if noise_type == "echo":
        distorted = _echo_distort(signal_f)
        return (np.clip(distorted, -1.0, 1.0) * 32768.0).astype(np.int16)

    # ── Additive types ────────────────────────────────────────────────────
    target_amp = _snr_amplitude(signal_f, snr_db)
    if target_amp == 0.0:
        return audio_int16  # silent input, skip noise

    if noise_type == "white":
        noise = _white_noise(n)
    elif noise_type == "pink":
        noise = _pink_noise(n)
    elif noise_type == "babble":
        noise = _babble_noise(n)
    else:  # office
        noise = _office_noise(n)

    noise_std = float(np.std(noise))
    if noise_std > 1e-10:
        noise *= target_amp / noise_std

    noisy = np.clip(signal_f + noise, -1.0, 1.0)
    return (noisy * 32768.0).astype(np.int16)


def save_noisy_wav(
    audio_int16: np.ndarray,
    noise_type: str,
    snr_db: float,
    output_path: Path,
) -> None:
    """Apply noise and write to a 16 kHz mono WAV file for auditing."""
    noisy = add_noise(audio_int16, noise_type, snr_db)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(noisy.astype("<i2").tobytes())
