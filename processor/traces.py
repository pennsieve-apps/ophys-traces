"""
Neuropil correction, baseline, ΔF/F, noise level, SNR and OASIS events.

- F_corr = F - r * F_neu (r = NEUROPIL_COEFFICIENT, 0.7 by default as in Suite2p and most labs).
- F0, the baseline of F_corr:
  - `maximin` (Suite2p's `dcnv_preprocess`): Gaussian smoothing (sigma 10 frames), then a
    running minimum and a running maximum, each over BASELINE_WINDOW_S;
  - `percentile`: a running BASELINE_PERCENTILE-th percentile over BASELINE_WINDOW_S (computed
    every window/8 and linearly interpolated).
  F0 is floored at 10 % of the ROI's median raw F (so a dim ROI whose neuropil-corrected
  baseline comes close to 0 does not produce huge ΔF/F); ROIs where the floor applies to more
  than 10 % of frames are flagged (`f0_floored`).
- ΔF/F = (F_corr - F0) / F0 (a fraction, not %).
- Noise level (CASCADE's definition, Rupprecht et al. 2021): median(|ΔF/F[t+1] - ΔF/F[t]|) /
  sqrt(frame rate) * 100. Comparable across frame rates; 1-2 is very clean, > 8 noisy.
- SNR: (99th percentile of ΔF/F - median) / sigma, sigma = the noise s.d. estimated from the
  frame-to-frame differences (1.4826 * median|diff| / sqrt(2)), robust to the activity itself.
- OASIS events: `oasis.functions.deconvolve` (AR(1) with the decay TAU) on each ΔF/F trace,
  with a minimal event size `s_min` = OASIS_THRESHOLD_SD x the noise s.d.: discrete events with
  amplitudes in ΔF/F units.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

FLOOR_FRACTION = 0.1


def neuropil_correct(F, Fneu, r: float) -> np.ndarray:
    return np.asarray(F, np.float64) - r * np.asarray(Fneu, np.float64)


def baseline_maximin(x: np.ndarray, fs: float, window_s: float, sigma_frames: float = 10.0) -> np.ndarray:
    win = max(3, int(round(window_s * fs)))
    s = gaussian_filter1d(x, sigma_frames, axis=1, mode="nearest")
    s = minimum_filter1d(s, win, axis=1, mode="nearest")
    return maximum_filter1d(s, win, axis=1, mode="nearest")


def baseline_percentile(x: np.ndarray, fs: float, window_s: float, q: float) -> np.ndarray:
    n, T = x.shape
    win = max(3, int(round(window_s * fs)))
    if win >= T:
        return np.repeat(np.percentile(x, q, axis=1, keepdims=True), T, axis=1)
    step = max(1, win // 8)
    centres = np.arange(0, T, step)
    if centres[-1] != T - 1:
        centres = np.append(centres, T - 1)
    vals = np.empty((n, len(centres)))
    half = win // 2
    for j, c in enumerate(centres):
        a, b = max(0, c - half), min(T, c + half + 1)
        vals[:, j] = np.percentile(x[:, a:b], q, axis=1)
    t = np.arange(T)
    return np.vstack([np.interp(t, centres, v) for v in vals])


def dff(F, Fneu, fs: float, r: float, method: str, window_s: float, q: float):
    """ΔF/F (rois x frames), F0, and per-ROI fraction of frames where F0 hit the floor."""
    Fc = neuropil_correct(F, Fneu, r)
    if method == "maximin":
        F0 = baseline_maximin(Fc, fs, window_s)
    else:
        F0 = baseline_percentile(Fc, fs, window_s, q)
    floor = FLOOR_FRACTION * np.maximum(np.median(np.asarray(F, np.float64), axis=1, keepdims=True), 1e-6)
    floored = (F0 < floor).mean(axis=1)
    F0 = np.maximum(F0, floor)
    return ((Fc - F0) / F0).astype(np.float32), F0.astype(np.float32), floored


def noise_level(d: np.ndarray, fs: float) -> np.ndarray:
    return np.nanmedian(np.abs(np.diff(d, axis=1)), axis=1) / np.sqrt(fs) * 100


def snr(d: np.ndarray) -> np.ndarray:
    sigma = 1.4826 * np.nanmedian(np.abs(np.diff(d, axis=1)), axis=1) / np.sqrt(2)
    sig = np.nanpercentile(d, 99, axis=1) - np.nanmedian(d, axis=1)
    return sig / np.maximum(sigma, 1e-9)


def noise_sd(d: np.ndarray) -> np.ndarray:
    """Per-ROI noise s.d. of ΔF/F from frame-to-frame differences (robust to the activity)."""
    return 1.4826 * np.nanmedian(np.abs(np.diff(d, axis=1)), axis=1) / np.sqrt(2)


def oasis_events(d: np.ndarray, fs: float, tau: float, threshold_sd: float) -> np.ndarray:
    """OASIS (AR(1) with decay tau) on each ΔF/F trace, keeping only events of at least
    threshold_sd x the trace's noise s.d. (OASIS's `s_min`, the thresholded variant of Friedrich,
    Zhou & Paninski 2017): event amplitudes in ΔF/F units, 0 elsewhere."""
    from oasis.functions import deconvolve
    out = np.zeros_like(d, dtype=np.float32)
    sds = noise_sd(d)
    for i, y in enumerate(np.asarray(d, np.float64)):
        if not np.all(np.isfinite(y)) or np.ptp(y) == 0:
            continue
        try:
            res = deconvolve(y, tau_d=tau, framerate=fs, penalty=None, s_min=float(threshold_sd * sds[i]))
        except Exception:
            continue
        out[i] = np.asarray(res.s, np.float32)
    return out
