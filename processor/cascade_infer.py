"""
CASCADE spike inference with CascadeTorch (the PyTorch port of CASCADE by Peter Rupprecht), using
the pretrained models baked into the image (CASCADE_MODELS, default /opt/cascade/Pretrained_models).

Model choice (CASCADE_MODEL=auto), following CASCADE's own advice (a model trained at the
recording's frame rate, on ground truth resampled to its noise level):
1. the family: `GC8_EXC_*` (all GCaMP8 ground truth, Rupprecht et al. 2026) when the indicator is
   a GCaMP8 (INDICATOR, or the session NWB's imaging plane), else `Global_EXC_*` (the generalist
   excitatory-neuron models: GCaMP6, OGB-1, Cal-520, R-CaMP, ...);
2. the frame rate closest to the recording's (on a log scale);
3. the standard model when its training noise levels cover the 90th percentile of the session's
   noise levels, else the `_high_noise` model of that rate (trained up to noise level 14-45).
When the recording's rate differs from the model's by more than 5 %, ΔF/F is resampled to the
model's rate (linear interpolation), and the inferred rate is interpolated back.

CASCADE predicts the expected number of spikes per frame; this module returns spikes/s (x the
model's frame rate). The first and last half-window of frames (about 32 frames) have no
prediction and are NaN (`edge_frames` in the result).
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("processor")
MODELS = Path(os.environ.get("CASCADE_MODELS", "/opt/cascade/Pretrained_models"))
NAME_RE = re.compile(r"^(Global|GC8)_EXC_([0-9.]+)Hz_smoothing(\d+)ms(_high_noise)?$")


def available_models(folder: Path = MODELS) -> list[dict]:
    from ruamel.yaml import YAML
    out = []
    if not folder.is_dir():
        return out
    yaml = YAML(typ="safe")
    for d in sorted(folder.iterdir()):
        m = NAME_RE.match(d.name)
        cfg = d / "config.yaml"
        if not m or not cfg.is_file():
            continue
        c = yaml.load(cfg.read_text())
        out.append({"name": d.name, "family": m.group(1), "rate": float(c["sampling_rate"]),
                    "high_noise": bool(m.group(4)), "noise_levels": [float(x) for x in c["noise_levels"]],
                    "smoothing_s": float(c["smoothing"]), "window": int(c["windowsize"]),
                    "before_frac": float(c["before_frac"])})
    return out


def family_for(indicator: str | None, param: str) -> tuple[str, str]:
    """('GC8' | 'Global', why)."""
    if param == "gcamp8":
        return "GC8", "INDICATOR=gcamp8"
    if param in ("gcamp6f", "gcamp6s", "gcamp6", "other"):
        return "Global", f"INDICATOR={param}: Global_EXC models (GCaMP6 among their ground truth)"
    ind = (indicator or "").lower().replace(" ", "")
    if re.search(r"gcamp8|jgcamp8|gcamp-8", ind):
        return "GC8", f"indicator {indicator!r} from the session"
    if ind:
        return "Global", f"indicator {indicator!r} from the session"
    return "Global", "indicator unknown (INDICATOR=auto): generalist Global_EXC model"


def choose_model(models: list[dict], family: str, fs: float, noise_p90: float, override: str = "auto") -> tuple[dict, str]:
    if override and override != "auto":
        m = next((m for m in models if m["name"] == override), None)
        if m is None:
            raise ValueError(f"CASCADE_MODEL={override!r} is not bundled; available: "
                             f"{', '.join(m['name'] for m in models)}")
        return m, "CASCADE_MODEL"
    fam = [m for m in models if m["family"] == family] or models
    if not fam:
        raise RuntimeError(f"no CASCADE models in {MODELS}")
    best_rate = min({m["rate"] for m in fam}, key=lambda r: abs(math.log(r / fs)))
    at_rate = [m for m in fam if m["rate"] == best_rate]
    std = [m for m in at_rate if not m["high_noise"]]
    hi = [m for m in at_rate if m["high_noise"]]
    if std and (not hi or max(std[0]["noise_levels"]) >= noise_p90):
        return std[0], (f"{family} family, {best_rate:g} Hz (closest to {fs:.3g} Hz), noise p90 "
                        f"{noise_p90:.2f} within the standard model's {min(std[0]['noise_levels']):g}-"
                        f"{max(std[0]['noise_levels']):g}")
    m = hi[0] if hi else std[0]
    return m, (f"{family} family, {best_rate:g} Hz (closest to {fs:.3g} Hz), noise p90 {noise_p90:.2f}: "
               f"high-noise model (noise levels {min(m['noise_levels']):g}-{max(m['noise_levels']):g})")


def infer(dff: np.ndarray, times: np.ndarray, fs: float, model: dict, device: str = "cpu",
          max_bytes: int = 512 * 2**20, progress=None) -> dict:
    """CASCADE spike rate (spikes/s, rois x frames) for ΔF/F traces (fractions)."""
    import torch
    from cascade2p import cascade
    rate = model["rate"]
    resample = abs(fs - rate) / rate > 0.05
    if resample:
        t_model = np.arange(times[0], times[-1] + 0.5 / rate, 1.0 / rate)
        x = np.vstack([np.interp(t_model, times, d) for d in dff]).astype(np.float64)
    else:
        x = np.asarray(dff, np.float64)
    T = x.shape[1]
    # CASCADE materialises a rois x frames x window float64 array: chunk the ROIs to bound memory
    chunk = max(1, int(max_bytes // (T * model["window"] * 8 * 2)))
    out = np.full(x.shape, np.nan, dtype=np.float32)
    dev = torch.device(device)
    t0 = time.time()
    for a in range(0, x.shape[0], chunk):
        b = min(a + chunk, x.shape[0])
        y = cascade.predict(model["name"], x[a:b], model_folder=str(MODELS), threshold=0,
                            padding=np.nan, verbosity=0, device=dev)
        out[a:b] = y
        if progress and time.time() - t0 > 30:
            progress(b, x.shape[0])
            t0 = time.time()
    out *= rate          # expected spikes per model frame -> spikes per second
    if resample:
        out = np.vstack([np.interp(times, t_model, o) for o in out]).astype(np.float32)
    half = int(model["before_frac"] * model["window"])
    edge = int(math.ceil(half * fs / rate)) if resample else half
    return {"rate": out, "resampled": resample, "edge_frames": edge}


def noise_levels_for_model(dff: np.ndarray, fs: float) -> np.ndarray:
    """CASCADE's noise level (same as traces.noise_level, kept here for clarity)."""
    return np.nanmedian(np.abs(np.diff(dff, axis=1)), axis=1) / np.sqrt(fs) * 100
