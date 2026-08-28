#!/usr/bin/env python3
"""
patina.py — Analog coloration (tube / vinyl) for audio files.

Adds harmonic distortion and the artefacts of analog playback to FLAC (or
WAV/AIFF) files, processing whole albums and preserving tags and artwork.

WHY NOT A FILTER
----------------
A filter — FIR, IIR, biquad, anything linear and time-invariant — cannot
create frequencies that were not already in the signal. Feed it a 1 kHz sine
and you get a 1 kHz sine back, with a different amplitude and phase. Harmonics
require a *non-linearity*: a function y = f(x) applied sample by sample. That
is what this tool does, with anti-aliasing oversampling around it.

Typical use
-----------
  # A whole album, gentle single-ended tube model
  python3 patina.py "~/Music/Album" -o "~/Music/Album [tube]" --mode tube

  # Vinyl simulation, recursive over a whole library
  python3 patina.py ~/Music -o ~/Music_vinyl --mode vinyl --recursive

  # Cassette simulation, Dolby B tracking included
  python3 patina.py "~/Music/Album" -o "~/Music/Album [cassette]" --preset cassette

  # Measure the harmonic spectrum produced (writes no files)
  python3 patina.py --analyze --mode tube --drive 0.4

Models
------
  tube      Single-ended triode: tanh with an asymmetric bias -> 2nd harmonic
            dominance, growing with level, plus soft peak compression.
  vinyl     Turntable chain: bass summed to mono below a corner frequency,
            channel crosstalk, wow & flutter, tracking distortion (rising
            with frequency), HF roll-off and, optionally, the surface
            artefacts: rumble, hiss, clicks and pops.
  cassette  Tape chain: crosstalk, wow & flutter, tape saturation, bass-EQ
            bump, HF roll-off, head azimuth loss and, optionally, tape hiss
            (with simulated Dolby B/C tracking) and print-through.
  vinyl-tube     vinyl followed by tube (turntable -> valve preamp).
                 ('both' is a deprecated alias, from before cassette existed.)
  cassette-tube  cassette followed by tube (tape deck -> valve preamp).

Main parameters
---------------
  --drive       Strength of the non-linearity (0 = none, 1 = heavy).
  --bias        Waveshaper asymmetry: sets the 2nd/3rd harmonic ratio.
                0 = odd harmonics only, 0.2-0.4 = 2nd harmonic dominance.
  --mix         Percentage of processed signal (100 = fully wet).
  --oversample  Anti-aliasing oversampling factor (default 8).
  --tube-drive  In modes 'vinyl-tube' and 'cassette-tube', drive of the
  --tube-bias   valve stage that follows the whole vinyl/cassette chain
                (see 'vinyl-tube' and 'cassette-tube' in PRESETS).

MIT licence. Copyright (c) 2026 Carlo Mazzi. See LICENSE.
"""

import argparse
import os
import shutil
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly, butter, sosfilt, sosfiltfilt

try:
    import soundfile as sf
except ImportError:
    sys.exit("Missing 'soundfile': install with  pip install soundfile mutagen")

AUDIO_EXT = {".flac", ".wav", ".aiff", ".aif", ".w64", ".ogg"}


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

@dataclass
class Params:
    """Every parameter of the coloration chain."""
    mode: str = "tube"
    drive: float = 0.35          # non-linearity strength
    bias: float = 0.25           # asymmetry -> even harmonics
    mix: float = 100.0           # % wet
    oversample: int = 8

    # --- vinyl ---
    bass_mono_hz: float = 150.0  # side channel removed below this frequency
    crosstalk_db: float = -30.0  # channel bleed
    wow_pct: float = 0.12        # % of slow pitch modulation (0.55 Hz)
    flutter_pct: float = 0.03    # % of fast pitch modulation (8-10 Hz)
    hf_rolloff_db: float = -1.5  # top-end shelf amount (vinyl and cassette)
    hf_corner_hz: float = 10000.0  # top-end shelf corner (vinyl and cassette)
    tilt_db: float = 6.0         # HF pre-emphasis: frequency-dependent drive
    rumble_db: float = -80.0     # turntable rumble level (-999 = off)
    noise_db: float = -999.0     # surface hiss (-999 = off)
    click_rate: float = 0.0      # clicks per second (0 = off)
    click_db: float = -42.0      # click peak level
    tick_db: float = -999.0      # periodic pop, once per revolution (-999 = off)
    rpm: float = 33.333          # platter speed, for the periodic pop

    # --- cassette ---
    hiss_db: float = -999.0      # tape hiss, broadband and top-heavy (-999 = off)
    lf_bump_hz: float = 90.0     # head-bump / bass-EQ corner
    lf_bump_db: float = 0.0      # boost below lf_bump_hz (0 = flat)
    azimuth_db: float = 0.0      # extra HF loss from head misalignment (0 = off)
    azimuth_corner_hz: float = 6000.0
    dolby_type: str = "off"      # 'off', 'b' or 'c': simulated NR tracking
    dolby_mismatch_pct: float = 15.0  # 0 = perfect tracking, 100 = no reduction
    print_db: float = -999.0     # print-through / pre-echo level (-999 = off)
    print_ms: float = 150.0      # how far ahead of the sound it is heard

    # --- second valve stage (modes 'vinyl-tube' / 'cassette-tube' only) ---
    tube_drive: float = 0.30     # drive of the valve stage after the vinyl/cassette chain
    tube_bias: float = 0.28      # asymmetry of that stage

    # --- output ---
    headroom_db: float = 0.5     # margin below 0 dBFS
    match_rms: bool = True       # match the gain to the source RMS
    dither: bool = True          # TPDF dither on 16-bit output


PRESETS = {
    "tube-soft":  dict(mode="tube", drive=0.20, bias=0.20),
    "tube":       dict(mode="tube", drive=0.35, bias=0.25),
    "tube-hot":   dict(mode="tube", drive=0.60, bias=0.35),
    "vinyl-mint": dict(mode="vinyl", drive=0.15, bias=0.25, wow_pct=0.06,
                       flutter_pct=0.02, crosstalk_db=-32.0),
    "vinyl":      dict(mode="vinyl", drive=0.30, bias=0.30),
    "vinyl-worn": dict(mode="vinyl", drive=0.45, bias=0.35, wow_pct=0.25,
                       flutter_pct=0.08, crosstalk_db=-22.0, noise_db=-62.0,
                       rumble_db=-70.0, click_rate=4.0, click_db=-42.0),
    "vinyl-trashed": dict(mode="vinyl", drive=0.55, bias=0.35, wow_pct=0.40,
                          flutter_pct=0.14, crosstalk_db=-18.0,
                          noise_db=-54.0, rumble_db=-62.0, click_rate=14.0,
                          click_db=-34.0, tick_db=-30.0, hf_rolloff_db=-3.5),
    "vinyl-tube": dict(mode="vinyl-tube", drive=0.30, bias=0.30,
                       tube_drive=0.30, tube_bias=0.28),
    "cassette-chrome": dict(mode="cassette", drive=0.15, bias=0.10,
                       hiss_db=-58.0, hf_corner_hz=15000.0, hf_rolloff_db=-2.0,
                       lf_bump_hz=60.0, lf_bump_db=0.5, wow_pct=0.08,
                       flutter_pct=0.06, crosstalk_db=-38.0,
                       dolby_type="b", dolby_mismatch_pct=5.0),
    "cassette":   dict(mode="cassette", drive=0.25, bias=0.15,
                       hiss_db=-50.0, hf_corner_hz=11000.0, hf_rolloff_db=-5.0,
                       lf_bump_hz=90.0, lf_bump_db=1.5, wow_pct=0.15,
                       flutter_pct=0.12, crosstalk_db=-35.0,
                       dolby_type="b", dolby_mismatch_pct=15.0),
    "cassette-worn": dict(mode="cassette", drive=0.40, bias=0.20,
                       hiss_db=-42.0, hf_corner_hz=8000.0, hf_rolloff_db=-9.0,
                       lf_bump_hz=100.0, lf_bump_db=2.5, wow_pct=0.35,
                       flutter_pct=0.30, crosstalk_db=-28.0, azimuth_db=-4.0,
                       azimuth_corner_hz=5000.0, dolby_type="b",
                       dolby_mismatch_pct=40.0, print_db=-38.0, print_ms=160.0),
    "cassette-no-dolby": dict(mode="cassette", drive=0.30, bias=0.18,
                       hiss_db=-46.0, hf_corner_hz=12000.0, hf_rolloff_db=-4.0,
                       lf_bump_hz=85.0, lf_bump_db=1.5, wow_pct=0.15,
                       flutter_pct=0.12, crosstalk_db=-33.0, dolby_type="off"),
    "cassette-tube": dict(mode="cassette-tube", drive=0.25, bias=0.15,
                       hiss_db=-50.0, hf_corner_hz=11000.0, hf_rolloff_db=-5.0,
                       lf_bump_hz=90.0, lf_bump_db=1.5, wow_pct=0.15,
                       flutter_pct=0.12, crosstalk_db=-35.0, dolby_type="b",
                       dolby_mismatch_pct=15.0, tube_drive=0.30, tube_bias=0.28),
}

PRESETS["both"] = PRESETS["vinyl-tube"]  # deprecated alias, pre-cassette name


# --------------------------------------------------------------------------
# DSP building blocks
# --------------------------------------------------------------------------

def waveshape(x: np.ndarray, drive: float, bias: float) -> np.ndarray:
    """Asymmetric single-ended triode non-linearity.

        y = tanh(k*(x + b)) - tanh(k*b)

    The constant term cancels the DC offset introduced by the bias. With
    b > 0 the function loses its odd symmetry and generates EVEN harmonics
    (the 2nd above all), whose amplitude grows with signal level — exactly
    what a triode stage does. With b = 0 it reduces to a plain tanh: odd
    harmonics only, the "solid state" character.
    """
    if drive <= 0.0:
        return x
    # Mapping calibrated on a -6 dBFS sine (bias 0.25):
    #   drive 0.2 -> ~0.6% THD | 0.35 -> ~1.8% | 0.6 -> ~5% | 1.0 -> ~11%
    k = 1.6 * drive
    # normalise the small-signal slope, so that the gain stays at ~1
    slope = k * (1.0 - np.tanh(k * bias) ** 2)
    # In-place arithmetic: in the oversampled domain each array is over 1 GB,
    # and intermediate copies would push the machine into swap.
    y = np.add(x, bias)
    y *= k
    np.tanh(y, out=y)
    y -= np.tanh(k * bias)
    y /= slope
    return y


def tilt_filter(x: np.ndarray, fs: float, db: float, corner: float = 3000.0):
    """First-order shelf: +db at the top end (pass -db for the inverse)."""
    if abs(db) < 1e-6:
        return x
    g = 10.0 ** (db / 20.0)
    w = np.tan(np.pi * corner / fs)
    c = (1.0 - w) / (1.0 + w)
    # out = lo + (x - lo)*g = x*g + lo*(1-g), computed in place
    lo = _onepole_lp(x, c)
    lo *= (1.0 - g)
    out = x * g
    out += lo
    return out


def _onepole_lp(x: np.ndarray, c: float) -> np.ndarray:
    """One-pole low-pass, vectorised through lfilter."""
    from scipy.signal import lfilter
    b = [(1.0 - c) / 2.0, (1.0 - c) / 2.0]
    a = [1.0, -c]
    return lfilter(b, a, x, axis=0)


def dc_block(x: np.ndarray, fs: float, fc: float = 5.0) -> np.ndarray:
    sos = butter(2, fc / (fs / 2.0), btype="highpass", output="sos")
    return sosfilt(sos, x, axis=0)


def bass_to_mono(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Sum the content below fc to mono (a physical limit of disc cutting)."""
    if x.ndim < 2 or x.shape[1] < 2 or fc <= 0:
        return x
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    sos = butter(2, fc / (fs / 2.0), btype="highpass", output="sos")
    side = sosfiltfilt(sos, side)          # linear phase: no smearing
    out = np.empty_like(x)
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return out


def crosstalk(x: np.ndarray, db: float) -> np.ndarray:
    """Channel bleed: about -30 dB on a good turntable."""
    if x.ndim < 2 or x.shape[1] < 2 or db <= -90.0:
        return x
    c = 10.0 ** (db / 20.0)
    out = np.empty_like(x)
    out[:, 0] = x[:, 0] + c * x[:, 1]
    out[:, 1] = x[:, 1] + c * x[:, 0]
    return out / (1.0 + c)


def lf_shelf_filter(x: np.ndarray, fs: float, db: float, corner: float = 90.0):
    """First-order low shelf: +db below corner, unity above (a bass EQ / head
    bump). The mirror image of tilt_filter, which shelves the top end."""
    if abs(db) < 1e-6:
        return x
    g = 10.0 ** (db / 20.0)
    w = np.tan(np.pi * corner / fs)
    c = (1.0 - w) / (1.0 + w)
    lo = _onepole_lp(x, c)
    hi = x - lo
    lo *= g
    lo += hi
    return lo


def azimuth_loss(x: np.ndarray, fs: float, db: float, corner: float = 6000.0):
    """Head azimuth misalignment: extra high-frequency loss, worse on one
    channel than the other because consumer decks are rarely perfectly
    aligned. A true azimuth error is a few microseconds of inter-channel
    delay, comb-filtering the top end away; this is a cheaper stand-in with
    the same audible result — duller, narrower stereo highs."""
    if x.ndim < 2 or x.shape[1] < 2 or db >= 0:
        return x
    out = np.empty_like(x)
    out[:, 0] = tilt_filter(x[:, 0], fs, db * 0.4, corner=corner)
    out[:, 1] = tilt_filter(x[:, 1], fs, db, corner=corner)
    return out


def tape_hiss(y: np.ndarray, fs: float, hiss_db: float, dolby_type: str,
             mismatch_pct: float, rng: np.random.Generator) -> np.ndarray:
    """Tape hiss, optionally run through a simulated Dolby B/C tracking.

    Real Dolby is a sliding-band compander: quiet high-frequency content is
    boosted going onto the tape and pulled back down on playback by the same
    amount, taking the hiss picked up in that pass down with it whenever the
    programme is loud. A decoder that tracks perfectly is inaudible; get the
    type wrong, or the calibration slightly off, and the residual
    cancellation error is heard as the noise floor "breathing" in time with
    the music — `mismatch_pct` sets how much of the ideal tracking is missed
    (0 = perfect and inaudible, 100 = no reduction at all).
    """
    if hiss_db <= -200.0:
        return np.zeros_like(y)
    n, nch = y.shape
    amp = 10.0 ** (hiss_db / 20.0)
    hiss = rng.standard_normal((n, nch))
    sos = butter(1, 1200.0 / (fs / 2.0), btype="highpass", output="sos")
    hiss = amp * sosfilt(sos, hiss, axis=0)

    if dolby_type == "off":
        return hiss

    depth = 0.6 if dolby_type == "b" else 0.85  # 'c' tracks a wider band, deeper
    sos_hf = butter(2, 2500.0 / (fs / 2.0), btype="highpass", output="sos")
    hf = sosfilt(sos_hf, y, axis=0)
    env = np.abs(hf).mean(axis=1)
    c = np.exp(-1.0 / (0.020 * fs))          # ~20 ms follower, Dolby's own ballpark
    env = _onepole_lp(env, c)
    peak = np.max(env) or 1.0
    env /= peak

    ideal = 1.0 / (1.0 + depth * env)        # decoder gain: down when the music is loud
    miss = np.clip(mismatch_pct, 0.0, 100.0) / 100.0
    gain = 1.0 + (ideal - 1.0) * (1.0 - miss)  # imperfect cancellation
    return hiss * gain[:, None]


def print_through(y: np.ndarray, fs: float, db: float,
                  delay_ms: float) -> np.ndarray:
    """Print-through: a faint, muffled copy of a loud passage transferred
    magnetically onto the adjacent tape layer, heard *before* the passage
    itself. Pre-echo dominates over post-echo because forward masking is
    much weaker than backward masking, so only the pre-echo is modelled."""
    if db <= -200.0 or delay_ms <= 0:
        return np.zeros_like(y)
    d = int(delay_ms * 1e-3 * fs)
    if d <= 0 or d >= y.shape[0]:
        return np.zeros_like(y)
    amp = 10.0 ** (db / 20.0)
    sos = butter(2, 4000.0 / (fs / 2.0), btype="lowpass", output="sos")
    lp = sosfilt(sos, y, axis=0)
    echo = np.zeros_like(y)
    echo[:-d] = lp[d:] * amp
    return echo


def wow_flutter(x: np.ndarray, fs: float, wow_pct: float,
                flutter_pct: float, rng: np.random.Generator) -> np.ndarray:
    """Pitch modulation: slow wow (~0.55 Hz, one platter revolution) + flutter.

    Implemented as resampling at modulated time positions, i.e. a delay line
    of varying length. It belongs in the oversampled domain, where linear
    interpolation is harmless.

    Processing runs in blocks: at 8x a four-minute track reaches 85 M samples
    per channel, and full-length temporaries would send the machine into swap.
    """
    if wow_pct <= 0 and flutter_pct <= 0:
        return x
    n = x.shape[0]
    if n < 2:
        return x

    # initial phases, fixed once so that the blocks stay coherent
    ph_w1, ph_w2, ph_f = rng.uniform(0, 2 * np.pi, 3)
    out = np.empty_like(x)
    acc = 0.0          # current read position, in samples
    offset = None      # alignment: playback must start at sample 0
    chunk = 1 << 20

    for s0 in range(0, n, chunk):
        s1 = min(n, s0 + chunk)
        t = np.arange(s0, s1, dtype=np.float64) / fs
        dev = np.zeros(s1 - s0, dtype=np.float64)
        if wow_pct > 0:
            dev += (wow_pct / 100.0) * np.sin(2 * np.pi * 0.55 * t + ph_w1)
            dev += (wow_pct / 200.0) * np.sin(2 * np.pi * 1.70 * t + ph_w2)
        if flutter_pct > 0:
            dev += (flutter_pct / 100.0) * np.sin(2 * np.pi * 9.30 * t + ph_f)
        del t

        # read position = integral of the speed deviation
        dev += 1.0
        pos = np.cumsum(dev)
        pos += acc
        acc = pos[-1]
        if offset is None:
            offset = pos[0]
        pos -= offset
        np.clip(pos, 0.0, n - 1.0, out=pos)

        i0 = pos.astype(np.int64)
        frac = pos - i0
        del pos
        i1 = np.minimum(i0 + 1, n - 1)
        for ch in range(x.shape[1]):
            col = x[:, ch]
            out[s0:s1, ch] = col[i0] * (1.0 - frac) + col[i1] * frac

    return out


def surface_noise(n: int, nch: int, fs: float, noise_db: float,
                  rumble_db: float, rng: np.random.Generator) -> np.ndarray:
    """Surface hiss plus turntable rumble."""
    out = np.zeros((n, nch))
    if noise_db > -200.0:
        amp = 10.0 ** (noise_db / 20.0)
        hiss = rng.standard_normal((n, nch))
        sos = butter(1, 2000.0 / (fs / 2.0), btype="highpass", output="sos")
        out += amp * sosfilt(sos, hiss, axis=0)
    if rumble_db > -200.0:
        amp = 10.0 ** (rumble_db / 20.0)
        low = rng.standard_normal((n, nch))
        sos = butter(2, 45.0 / (fs / 2.0), btype="lowpass", output="sos")
        low = sosfilt(sos, low, axis=0)
        peak = np.max(np.abs(low)) or 1.0
        out += amp * low / peak
    return out


def _click_template(fs: float, tau: float, f_lo: float, f_hi: float,
                    rng: np.random.Generator) -> np.ndarray:
    """One click: band-limited noise under an exponentially decaying envelope.

    The passband sets the character — narrow and high for fine crackle, lower
    and longer for the thud of a deep scratch.
    """
    n = max(8, int(6.0 * tau * fs))
    t = np.arange(n) / fs
    sos = butter(2, [f_lo / (fs / 2), min(f_hi, fs / 2 * 0.98) / (fs / 2)],
                 btype="bandpass", output="sos")
    h = sosfilt(sos, rng.standard_normal(n))
    h *= np.exp(-t / tau)
    h[0] *= 3.0                       # sharp leading edge
    peak = np.max(np.abs(h))
    return h / peak if peak > 0 else h


def clicks(n: int, nch: int, fs: float, rate: float, level_db: float,
           tick_db: float, rpm: float, rng: np.random.Generator) -> np.ndarray:
    """Clicks, crackle and pops from the surface of the disc.

    Events follow a Poisson process: the intervals between one click and the
    next are random and independent, as they are for defects scattered over
    the surface. One event in twenty is a lower, louder "pop", the signature
    of a scratch. The level is absolute, not relative to the music: clicks
    stay masked in loud passages and emerge in quiet ones, exactly as they do
    on a record.
    """
    out = np.zeros((n, nch))
    if n <= 0:
        return out

    # families of waveforms, reused at random amplitudes and positions
    fine = [_click_template(fs, rng.uniform(0.2e-3, 0.9e-3),
                            1500.0, 9000.0, rng) for _ in range(8)]
    pops = [_click_template(fs, rng.uniform(2.0e-3, 6.0e-3),
                            300.0, 3500.0, rng) for _ in range(4)]

    def scatter(pos: int, tpl: np.ndarray, amp: float) -> None:
        L = min(len(tpl), n - pos)
        if L <= 0:
            return
        for ch in range(nch):
            # the same defect is heard on both channels, at slightly
            # different amplitudes and delays
            g = amp * (1.0 if ch == 0 else rng.uniform(0.6, 1.0))
            d = 0 if ch == 0 else int(rng.integers(0, 24))
            LL = min(L, n - pos - d)
            if LL > 0:
                out[pos + d: pos + d + LL, ch] += g * tpl[:LL]

    if rate > 0 and level_db > -200.0:
        amp0 = 10.0 ** (level_db / 20.0)
        n_ev = int(rng.poisson(rate * n / fs))
        for pos in rng.integers(0, n, n_ev):
            if rng.random() < 0.05:
                scatter(int(pos), pops[rng.integers(len(pops))],
                        amp0 * rng.uniform(1.5, 3.5))
            else:
                scatter(int(pos), fine[rng.integers(len(fine))],
                        amp0 * rng.uniform(0.25, 1.0))

    if tick_db > -200.0 and rpm > 0:
        # a fixed defect: it comes back on every revolution
        step = fs * 60.0 / rpm
        amp0 = 10.0 ** (tick_db / 20.0)
        tpl = pops[0]
        pos = rng.uniform(0, step)
        while pos < n:
            scatter(int(pos), tpl, amp0 * rng.uniform(0.8, 1.2))
            pos += step

    return out


# --------------------------------------------------------------------------
# The full chain
# --------------------------------------------------------------------------

def process(x: np.ndarray, fs: float, p: Params, seed: int = 0) -> np.ndarray:
    """Run the coloration chain over a block of (n_samples, n_channels)."""
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(seed)
    dry = x.copy()
    y = x.astype(np.float64, copy=True)

    rms_in = float(np.sqrt(np.mean(y ** 2))) if p.match_rms else 0.0

    # --- base rate pre-processing (vinyl / cassette) ---
    if p.mode in ("vinyl", "vinyl-tube"):
        y = bass_to_mono(y, fs, p.bass_mono_hz)
    if p.mode in ("vinyl", "vinyl-tube", "cassette", "cassette-tube"):
        y = crosstalk(y, p.crosstalk_db)

    # --- oversampled domain: wow/flutter and the non-linearity ---
    os_factor = max(1, int(p.oversample))
    fs_os = fs * os_factor
    if os_factor > 1:
        y = resample_poly(y, os_factor, 1, axis=0)

    if p.mode in ("vinyl", "vinyl-tube", "cassette", "cassette-tube"):
        y = wow_flutter(y, fs_os, p.wow_pct, p.flutter_pct, rng)

    if p.drive > 0:
        if p.mode in ("vinyl", "vinyl-tube") and p.tilt_db > 0:
            # tracking distortion rises with frequency:
            # pre-emphasis -> waveshaper -> de-emphasis
            y = tilt_filter(y, fs_os, +p.tilt_db)
            y = waveshape(y, p.drive, p.bias)
            y = tilt_filter(y, fs_os, -p.tilt_db)
        else:
            y = waveshape(y, p.drive, p.bias)

    if os_factor > 1:
        y = resample_poly(y, 1, os_factor, axis=0)
    y = y[:x.shape[0]] if y.shape[0] >= x.shape[0] else \
        np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0)))

    # --- base rate post-processing ---
    y = dc_block(y, fs)

    if p.mode in ("vinyl", "vinyl-tube", "cassette", "cassette-tube") \
            and p.hf_rolloff_db < 0:
        y = tilt_filter(y, fs, p.hf_rolloff_db, corner=p.hf_corner_hz)

    if p.mode in ("cassette", "cassette-tube"):
        if p.lf_bump_db != 0.0:
            y = lf_shelf_filter(y, fs, p.lf_bump_db, corner=p.lf_bump_hz)
        if p.azimuth_db < 0.0:
            y = azimuth_loss(y, fs, p.azimuth_db, corner=p.azimuth_corner_hz)

    # --- gain compensation ---
    # This has to happen on the musical programme alone: noise and clicks are
    # additive and set at an absolute level, so including them here would turn
    # the music down in proportion to how worn the record is meant to sound.
    if p.match_rms:
        rms_out = float(np.sqrt(np.mean(y ** 2)))
        if rms_out > 1e-12:
            y *= rms_in / rms_out

    # --- additive surface artefacts ---
    if p.mode in ("vinyl", "vinyl-tube"):
        y = y + surface_noise(y.shape[0], y.shape[1], fs,
                              p.noise_db, p.rumble_db, rng)
        y = y + clicks(y.shape[0], y.shape[1], fs, p.click_rate, p.click_db,
                       p.tick_db, p.rpm, rng)

    if p.mode in ("cassette", "cassette-tube"):
        y = y + tape_hiss(y, fs, p.hiss_db, p.dolby_type,
                          p.dolby_mismatch_pct, rng)
        y = y + print_through(y, fs, p.print_db, p.print_ms)

    # --- valve stage, downstream of the whole turntable/tape chain ---
    # This is the physical order: the cartridge/tape-head output, hiss and
    # clicks included, is what reaches the preamp, so the valve stage
    # colours those too. It needs its own oversampled round: the additive
    # artefacts are generated at base rate, after the first one.
    if p.mode in ("vinyl-tube", "cassette-tube") and p.tube_drive > 0:
        rms_pre = float(np.sqrt(np.mean(y ** 2)))
        if os_factor > 1:
            y = resample_poly(y, os_factor, 1, axis=0)
        y = waveshape(y, p.tube_drive, p.tube_bias)
        if os_factor > 1:
            y = resample_poly(y, 1, os_factor, axis=0)
        y = y[:x.shape[0]] if y.shape[0] >= x.shape[0] else \
            np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0)))
        y = dc_block(y, fs)
        # Unity gain through the non-linearity. Measured across this stage
        # alone, so the absolute level of the surface artefacts is preserved
        # (they sit 40 dB or more below the music and barely move the RMS).
        rms_post = float(np.sqrt(np.mean(y ** 2)))
        if p.match_rms and rms_post > 1e-12:
            y *= rms_pre / rms_post

    # --- mix ---
    w = np.clip(p.mix / 100.0, 0.0, 1.0)
    y = w * y + (1.0 - w) * dry

    # --- peak ceiling ---
    ceiling = 10.0 ** (-p.headroom_db / 20.0)
    peak = float(np.max(np.abs(y)))
    if peak > ceiling:
        y *= ceiling / peak
    return y


# --------------------------------------------------------------------------
# File I/O and tags
# --------------------------------------------------------------------------

def copy_tags(src: Path, dst: Path, peak: Optional[float] = None) -> str:
    """Copy tags and artwork from the source file to the processed one.

    If the file carries ReplayGain tags, the *_PEAK fields are updated with
    the real peak of the processed file. The gain value stays valid because
    the chain preserves the RMS level of the original.
    """
    try:
        from mutagen.flac import FLAC, Picture
    except ImportError:
        return "mutagen missing: tags not copied"

    if src.suffix.lower() != ".flac" or dst.suffix.lower() != ".flac":
        return "tags are only copied between FLAC files"

    try:
        s = FLAC(str(src))
        d = FLAC(str(dst))
        d.delete()
        for key, val in s.tags:
            if peak is not None and key.upper().endswith("_PEAK"):
                val = f"{peak:.6f}"
            d.tags.append((key, val))
        d.clear_pictures()
        for pic in s.pictures:
            new = Picture()
            new.data = pic.data
            new.type = pic.type
            new.mime = pic.mime
            new.desc = pic.desc
            new.width, new.height = pic.width, pic.height
            new.depth, new.colors = pic.depth, pic.colors
            d.add_picture(new)
        d.save()
        return "ok"
    except Exception as exc:                       # pragma: no cover
        return f"tag error: {exc}"


def apply_dither(y: np.ndarray, bits: int, rng: np.random.Generator):
    """1 LSB TPDF dither, for 16-bit output."""
    lsb = 2.0 ** -(bits - 1)
    n = rng.random(y.shape) - rng.random(y.shape)   # triangular
    return y + n * lsb


def process_file(src: Path, dst: Path, p: Params, force: bool) -> tuple:
    """Process a single file. Returns (src, status, detail)."""
    if dst.exists() and not force:
        return (src, "skip", "already exists (use --force)")
    try:
        info = sf.info(str(src))
        x, fs = sf.read(str(src), dtype="float64", always_2d=True)
    except Exception as exc:
        return (src, "error", f"read: {exc}")

    # Seed derived from the file name, so the same track always gets the same
    # wow, flutter and clicks. It has to be a stable hash: the built-in hash()
    # is randomised per process, which would make every run — and every worker
    # in the pool — produce different artefacts.
    seed = zlib.crc32(src.name.encode("utf-8"))
    y = process(x, fs, p, seed=seed)

    subtype = info.subtype
    if src.suffix.lower() == ".flac" and subtype not in (
            "PCM_16", "PCM_24", "PCM_S8"):
        subtype = "PCM_24"
    if p.dither and subtype == "PCM_16":
        y = apply_dither(y, 16, np.random.default_rng(seed))
        y = np.clip(y, -1.0, 1.0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        sf.write(str(dst), y, int(fs), subtype=subtype)
    except Exception as exc:
        return (src, "error", f"write: {exc}")

    tag_status = copy_tags(src, dst, peak=float(np.max(np.abs(y))))
    return (src, "ok", f"{info.channels}ch {fs} Hz {subtype} | tags: {tag_status}")


def total_ram_bytes() -> int:
    """Physical RAM of the machine (falls back to a conservative 8 GB)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 8 << 30


def auto_jobs(files: list, p: Params, cap: int = 4) -> int:
    """How many parallel workers the available RAM can take.

    In the oversampled domain a four-minute track at 8x takes about 1.5 GB per
    array, and the chain needs roughly two arrays live per worker. The factor
    is calibrated on real measurements (a 42-minute album on a 17 GB machine):
    without this limit the run goes into swap and ends up slower than serial.
    """
    peak = 0
    for f in files[:200]:
        try:
            i = sf.info(str(f))
        except Exception:
            continue
        peak = max(peak, int(i.frames) * max(1, i.channels))
    if peak == 0:
        return 1
    per_job = peak * 8 * max(1, p.oversample) * 1.6
    budget = int(total_ram_bytes() * 0.55)
    n = max(1, int(budget // per_job))
    return int(min(cap, os.cpu_count() or 1, n, len(files)))


def collect_files(root: Path, recursive: bool) -> list:
    if root.is_file():
        return [root]
    it = root.rglob("*") if recursive else root.glob("*")
    # "._Name.ext" AppleDouble sidecars: macOS writes these on non-HFS+
    # volumes (network shares, exFAT, ...) to hold the resource fork /
    # extended attributes. They carry the same extension as the real file
    # but are not audio, so soundfile fails to open them.
    return sorted(f for f in it
                  if f.is_file() and f.suffix.lower() in AUDIO_EXT
                  and not f.name.startswith("._"))


EXTRA_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",   # artwork
             ".pdf", ".cue", ".log", ".txt", ".m3u", ".m3u8", ".nfo"}


def copy_extras(src_root: Path, dst_root: Path, files: list) -> int:
    """Copy the album's companion files next to the tracks.

    Loose artwork, booklets, cue sheets and rip logs. The cue sheet stays
    valid because the track file names do not change.
    """
    if src_root.is_file():
        src_root = src_root.parent
    n = 0
    dirs = {f.parent for f in files}
    for d in dirs:
        rel = d.relative_to(src_root) if d != src_root else Path(".")
        for img in d.glob("*"):
            if img.suffix.lower() in EXTRA_EXT and not img.name.startswith("._"):
                out = dst_root / rel / img.name
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.exists():
                    shutil.copy2(img, out)
                    n += 1
    return n


# --------------------------------------------------------------------------
# Harmonic analysis
# --------------------------------------------------------------------------

def analyze(p: Params, fs: int = 44100, f0: float = 1000.0,
            level_db: float = -6.0, n_harm: int = 8) -> None:
    """Measure the harmonic spectrum the chain produces from a sine wave."""
    # The frequency is locked to the FFT grid (k whole periods in n samples):
    # no leakage, no window needed. The period is deliberately NOT a whole
    # number of samples — otherwise the aliasing products would land exactly
    # on the harmonics and become invisible to the measurement.
    n = 1 << 18
    k = max(1, round(f0 * n / fs))
    f0 = k * fs / n
    amp = 10.0 ** (level_db / 20.0)
    # twice the length is processed and only the second half analysed: the
    # first block absorbs the transients (the 5 Hz DC blocker takes ~0.5 s)
    pad = n
    t = np.arange(n + pad) / fs
    x = (amp * np.sin(2 * np.pi * f0 * t))[:, None]

    q = Params(**{**asdict(p), "match_rms": False, "headroom_db": 0.0,
                  "wow_pct": 0.0, "flutter_pct": 0.0,
                  "noise_db": -999.0, "rumble_db": -999.0,
                  "hiss_db": -999.0, "print_db": -999.0})
    y = process(x, fs, q)[pad:pad + n, 0]

    spec = np.abs(np.fft.rfft(y)) * 2.0 / n
    fund = spec[k]
    tube = (f" tube_drive={q.tube_drive} tube_bias={q.tube_bias}"
            if q.mode in ("vinyl-tube", "cassette-tube") else "")
    print(f"\nHarmonic analysis — mode={q.mode} drive={q.drive} "
          f"bias={q.bias}{tube} oversample={q.oversample}")
    print(f"Sine at {f0:.1f} Hz, {level_db:.1f} dBFS, fs={fs} Hz\n")
    print(f"  {'harmonic':<10}{'freq (Hz)':>11}{'dB rel':>10}{'%':>9}")
    print("  " + "-" * 40)
    print(f"  {'1 (fund.)':<10}{f0:>11.0f}{0.0:>10.1f}{100.0:>9.2f}")
    hsum = 0.0
    for h in range(2, n_harm + 1):
        kh = k * h
        if kh >= len(spec):
            break
        ratio = spec[kh] / fund if fund > 0 else 0.0
        hsum += ratio ** 2
        db = 20 * np.log10(ratio) if ratio > 1e-12 else -np.inf
        print(f"  {h:<10}{f0*h:>11.0f}{db:>10.1f}{ratio*100:>9.3f}")
    thd = np.sqrt(hsum) * 100
    # non-harmonic residue: everything that does not fall on a multiple of k
    mask = np.ones(len(spec), dtype=bool)
    for h in range(1, len(spec) // k + 1):
        mask[max(0, k * h - 2): k * h + 3] = False
    mask[:3] = False
    alias = np.sqrt(np.sum(spec[mask] ** 2)) / fund * 100
    print("  " + "-" * 40)
    print(f"  THD = {thd:.3f} %   |   non-harmonic residue (aliasing/noise) "
          f"= {alias:.4f} %")
    even = sum((spec[k*h]/fund)**2 for h in (2, 4, 6) if k*h < len(spec))
    odd = sum((spec[k*h]/fund)**2 for h in (3, 5, 7) if k*h < len(spec))
    if odd > 0:
        print(f"  even/odd ratio = {np.sqrt(even/odd):.2f} "
              f"({'EVEN dominant (tube-like)' if even > odd else 'ODD dominant'})")
    print()


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analog coloration (tube/vinyl) for FLAC and WAV files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available presets: " + ", ".join(PRESETS))
    p.add_argument("input", nargs="?", metavar="PATH",
                   help="File or directory to process")
    p.add_argument("-o", "--out", metavar="DIR",
                   help="Destination directory (default: <input>_<mode>)")
    p.add_argument("--preset", choices=list(PRESETS),
                   help="Starting preset; individual options override it")
    p.add_argument("--mode",
                   choices=["tube", "vinyl", "vinyl-tube", "cassette",
                            "cassette-tube", "both"],
                   help="Coloration model (default: tube). 'both' is a "
                        "deprecated alias for 'vinyl-tube'")

    g = p.add_argument_group("non-linearity")
    g.add_argument("--drive", type=float, metavar="0-1",
                   help="Distortion strength (default 0.35)")
    g.add_argument("--bias", type=float, metavar="0-1",
                   help="Asymmetry: 0 = odd harmonics only, "
                        "0.2-0.4 = 2nd harmonic dominance (default 0.25)")
    g.add_argument("--mix", type=float, metavar="PCT",
                   help="Percentage of processed signal (default 100). In "
                        "vinyl mode, values below 100 need wow and flutter "
                        "set to 0, otherwise the unmodulated dry signal "
                        "produces a chorus effect")
    g.add_argument("--oversample", type=int, metavar="N",
                   help="Anti-aliasing oversampling factor (default 8)")

    g = p.add_argument_group("vinyl")
    g.add_argument("--bass-mono-hz", type=float, metavar="HZ",
                   help="Sum to mono below this frequency (default 150)")
    g.add_argument("--crosstalk-db", type=float, metavar="DB",
                   help="Channel bleed (default -30)")
    g.add_argument("--wow-pct", type=float, metavar="PCT",
                   help="Slow pitch modulation (default 0.12)")
    g.add_argument("--flutter-pct", type=float, metavar="PCT",
                   help="Fast pitch modulation (default 0.03)")
    g.add_argument("--hf-rolloff-db", type=float, metavar="DB",
                   help="Top-end shelf amount, vinyl and cassette "
                        "(default -1.5)")
    g.add_argument("--hf-corner-hz", type=float, metavar="HZ",
                   help="Top-end shelf corner, vinyl and cassette "
                        "(default 10000)")
    g.add_argument("--tilt-db", type=float, metavar="DB",
                   help="HF pre-emphasis: distortion rising with frequency "
                        "(default 6)")
    g.add_argument("--rumble-db", type=float, metavar="DB",
                   help="Turntable rumble level (default -80, -999 = off)")
    g.add_argument("--noise-db", type=float, metavar="DB",
                   help="Surface hiss (default off)")
    g.add_argument("--click-rate", type=float, metavar="PER_SEC",
                   help="Clicks and crackle per second (default 0 = off). "
                        "1-3 = well kept record, 5-15 = heavily played")
    g.add_argument("--click-db", type=float, metavar="DB",
                   help="Click peak level (default -42)")
    g.add_argument("--tick-db", type=float, metavar="DB",
                   help="Periodic pop, once per platter revolution "
                        "(default off)")
    g.add_argument("--rpm", type=float, metavar="RPM",
                   help="Platter speed for the periodic pop (default 33.333)")

    g = p.add_argument_group("cassette")
    g.add_argument("--hiss-db", type=float, metavar="DB",
                   help="Tape hiss, broadband and top-heavy (default off)")
    g.add_argument("--lf-bump-hz", type=float, metavar="HZ",
                   help="Bass bump / head-EQ corner (default 90)")
    g.add_argument("--lf-bump-db", type=float, metavar="DB",
                   help="Bass bump amount below lf-bump-hz (default 0)")
    g.add_argument("--azimuth-db", type=float, metavar="DB",
                   help="Extra HF loss from head misalignment, worse on one "
                        "channel (default 0 = off)")
    g.add_argument("--azimuth-corner-hz", type=float, metavar="HZ",
                   help="Corner of the azimuth HF loss (default 6000)")
    g.add_argument("--dolby-type", choices=["off", "b", "c"],
                   help="Simulated noise-reduction tracking (default off)")
    g.add_argument("--dolby-mismatch-pct", type=float, metavar="PCT",
                   help="How far the simulated decoder tracking misses the "
                        "ideal (0 = perfect, 100 = no reduction; default 15)")
    g.add_argument("--print-db", type=float, metavar="DB",
                   help="Print-through / pre-echo level (default off)")
    g.add_argument("--print-ms", type=float, metavar="MS",
                   help="Pre-echo lead time (default 150)")

    g = p.add_argument_group("valve stage (modes 'vinyl-tube' and "
                             "'cassette-tube' only)")
    g.add_argument("--tube-drive", type=float, metavar="0-1",
                   help="Drive of the valve stage placed after the whole "
                        "vinyl or cassette chain (default 0.30, 0 = off)")
    g.add_argument("--tube-bias", type=float, metavar="0-1",
                   help="Asymmetry of that stage (default 0.28)")

    g = p.add_argument_group("output")
    g.add_argument("--headroom-db", type=float, metavar="DB",
                   help="Margin below 0 dBFS (default 0.5)")
    g.add_argument("--match-rms", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="Match the gain to the source RMS (default yes)")
    g.add_argument("--dither", action=argparse.BooleanOptionalAction,
                   default=None,
                   help="TPDF dither on 16-bit output (default yes)")

    g = p.add_argument_group("batch")
    g.add_argument("--recursive", action="store_true",
                   help="Descend into subdirectories")
    g.add_argument("--jobs", type=int, default=0, metavar="N",
                   help="Parallel workers (0 = automatic)")
    g.add_argument("--force", action="store_true",
                   help="Overwrite files that already exist")
    g.add_argument("--dry-run", action="store_true",
                   help="Only list what would be processed")
    g.add_argument("--analyze", action="store_true",
                   help="Print the harmonic spectrum for the chosen settings "
                        "and exit (processes no files)")
    g.add_argument("--analyze-freq", type=float, default=1000.0, metavar="HZ",
                   help="Test sine frequency (default 1000)")
    g.add_argument("--analyze-level", type=float, default=-6.0,
                   metavar="DBFS", help="Test sine level (default -6)")
    g.add_argument("--analyze-fs", type=int, default=44100, metavar="HZ",
                   help="Test sample rate (default 44100)")
    return p


def params_from_args(args) -> Params:
    p = Params()
    if args.preset:
        for k, v in PRESETS[args.preset].items():
            setattr(p, k, v)
    for field in asdict(p):
        v = getattr(args, field, None)
        if v is not None:
            setattr(p, field, v)
    if p.mode == "both":          # deprecated alias, pre-cassette name
        p.mode = "vinyl-tube"
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    p = params_from_args(args)

    if args.analyze:
        analyze(p, fs=args.analyze_fs, f0=args.analyze_freq,
                level_db=args.analyze_level)
        return 0

    if not args.input:
        build_parser().print_usage()
        return 2

    src_root = Path(args.input).expanduser().resolve()
    if not src_root.exists():
        print(f"No such path: {src_root}", file=sys.stderr)
        return 1

    if args.out:
        dst_root = Path(args.out).expanduser().resolve()
    else:
        base = src_root if src_root.is_dir() else src_root.parent
        dst_root = base.parent / f"{base.name}_{p.mode}"

    if dst_root == src_root:
        print("Destination is the same as the source: aborted.",
              file=sys.stderr)
        return 1

    files = collect_files(src_root, args.recursive)
    if not files:
        print(f"No audio files in {src_root}", file=sys.stderr)
        return 1

    base = src_root if src_root.is_dir() else src_root.parent
    jobs = [(f, dst_root / f.relative_to(base)) for f in files]

    print(f"Mode: {p.mode} | drive={p.drive} bias={p.bias} mix={p.mix}% "
          f"oversample={p.oversample}x")
    print(f"Source     : {src_root}")
    print(f"Destination: {dst_root}")
    print(f"Files found: {len(jobs)}\n")

    if args.dry_run:
        for s, d in jobs:
            print(f"  {s.name}  ->  {d}")
        return 0

    if args.jobs > 0:
        nproc = args.jobs
    else:
        nproc = auto_jobs(files, p)
        print(f"Parallel workers: {nproc} (automatic, "
              f"{total_ram_bytes() / (1 << 30):.0f} GB of RAM)\n")
    ok = skipped = failed = 0

    if nproc > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=nproc) as ex:
            futs = {ex.submit(process_file, s, d, p, args.force): s
                    for s, d in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                src, status, detail = fut.result()
                print(f"[{i}/{len(jobs)}] {status:<6} {src.name}  {detail}")
                ok += status == "ok"
                skipped += status == "skip"
                failed += status == "error"
    else:
        for i, (s, d) in enumerate(jobs, 1):
            src, status, detail = process_file(s, d, p, args.force)
            print(f"[{i}/{len(jobs)}] {status:<6} {src.name}  {detail}")
            ok += status == "ok"
            skipped += status == "skip"
            failed += status == "error"

    # companion files only make sense when processing a directory: for a
    # single track it would drag along the cue sheet and booklet of the album
    n_img = copy_extras(src_root, dst_root, files) if src_root.is_dir() else 0
    print(f"\nDone: {ok} | skipped: {skipped} | failed: {failed}"
          + (f" | companion files copied: {n_img}" if n_img else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
