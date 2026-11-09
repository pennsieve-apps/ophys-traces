"""
Figures per session (PNG, 150 dpi, session id in the title) with the captions the report shows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DPI = 150


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def example_traces(sid: str, out: Path, times, dff, cascade, oasis, snr, accepted, roi_labels,
                   n: int, model: str, window_s: float = 120.0) -> dict | None:
    if n <= 0 or dff.shape[0] == 0:
        return None
    plt = _plt()
    idx = np.where(accepted)[0] if accepted.any() else np.arange(dff.shape[0])
    idx = idx[np.argsort(-np.nan_to_num(snr[idx]))][:n]
    t = np.asarray(times)
    # the window with the most CASCADE spikes among the chosen ROIs
    if t[-1] - t[0] > window_s:
        dt = np.median(np.diff(t))
        w = int(window_s / dt)
        act = np.nansum(cascade[idx], axis=0)
        cs = np.concatenate([[0], np.cumsum(act)])
        starts = np.arange(0, len(t) - w, max(1, w // 4))
        a = int(starts[np.argmax(cs[starts + w] - cs[starts])]) if len(starts) else 0
        sl = slice(a, a + w)
    else:
        sl = slice(0, len(t))
    fig, axs = plt.subplots(len(idx), 1, figsize=(10, 1.25 * len(idx) + 1.0), sharex=True, squeeze=False)
    for ax, i in zip(axs[:, 0], idx):
        ax.plot(t[sl], dff[i, sl], color="k", lw=0.6)
        ev = np.where(oasis[i, sl] > 0)[0]
        if len(ev):
            top = np.nanmax(dff[i, sl])
            ax.plot(t[sl][ev], np.full(len(ev), top * 1.08 if top > 0 else 0.1), "|", color="tab:blue",
                    ms=5, mew=0.8)
        ax2 = ax.twinx()
        ax2.fill_between(t[sl], 0, cascade[i, sl], color="tab:red", alpha=0.35, lw=0)
        ax2.set_ylim(0, max(1.0, float(np.nanmax(cascade[idx][:, sl])) * 1.1))
        ax2.tick_params(axis="y", labelsize=7, colors="tab:red")
        ax.set_ylabel(f"{roi_labels[i]}\nSNR {snr[i]:.1f}", fontsize=7, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)
        for sp in ("top",):
            ax.spines[sp].set_visible(False)
    axs[-1, 0].set_xlabel("time (s, session clock)")
    fig.suptitle(f"{sid}: ΔF/F (black, left axis), CASCADE spike rate (red, spikes/s, right axis), "
                 f"OASIS events (blue ticks)", fontsize=9)
    fig.tight_layout()
    p = out / f"{sid}_example_traces.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return {"path": p.name, "caption": (
        f"Example ROIs of {sid}: the {len(idx)} accepted cells with the highest signal-to-noise ratio, "
        f"over the {t[sl][-1] - t[sl][0]:.0f} s with the most activity. Black: ΔF/F (the neuropil-corrected "
        "fluorescence relative to its running baseline). Red: the spike rate CASCADE infers "
        f"(model {model}; calibrated in spikes per second, so an isolated spike shows as a brief bump "
        "whose area is about one spike). Blue ticks: discrete events found by OASIS. Each calcium "
        "transient should have red underneath it; red without a transient points to noise.")}


def noise_hist(sid: str, out: Path, noise, accepted, model: dict | None) -> dict:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    finite = np.isfinite(noise)
    bins = np.linspace(0, max(10.0, float(np.nanpercentile(noise[finite], 99)) * 1.1 if finite.any() else 10), 40)
    ax.hist(noise[finite & ~accepted], bins=bins, color="0.7", label="other ROIs")
    ax.hist(noise[finite & accepted], bins=bins, color="tab:green", alpha=0.8, label="accepted cells")
    if model:
        lo, hi = min(model["noise_levels"]), max(model["noise_levels"])
        ax.axvspan(lo, hi, color="tab:red", alpha=0.08, label=f"trained noise range of {model['name']}")
    ax.set_xlabel("CASCADE noise level (% ΔF/F / √Hz)")
    ax.set_ylabel("ROIs")
    ax.legend(fontsize=7, frameon=False)
    med = np.nanmedian(noise[accepted]) if accepted.any() else np.nan
    ax.set_title(f"{sid}: noise levels (median of accepted cells {med:.2f})", fontsize=10)
    fig.tight_layout()
    p = out / f"{sid}_noise_levels.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return {"path": p.name, "caption": (
        f"Shot-noise level of each ROI's ΔF/F in {sid}, as CASCADE defines it (the median frame-to-frame "
        "change, normalised by the square root of the frame rate, so values compare across microscopes). "
        "Around 1-2 is a very clean recording, 3-5 typical, above 8 noisy. CASCADE was trained on ground "
        "truth resampled to the shaded range: ROIs outside it get less reliable spike rates. A noise level "
        "far above the rest often means a dim ROI or a ΔF/F baseline close to zero.")}


def snr_hist(sid: str, out: Path, snr, accepted) -> dict:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    finite = np.isfinite(snr)
    hi = float(np.nanpercentile(snr[finite], 99)) * 1.1 if finite.any() else 10
    bins = np.linspace(min(0, float(np.nanmin(snr[finite])) if finite.any() else 0), max(hi, 5), 40)
    ax.hist(snr[finite & ~accepted], bins=bins, color="0.7", label="other ROIs")
    ax.hist(snr[finite & accepted], bins=bins, color="tab:green", alpha=0.8, label="accepted cells")
    ax.set_xlabel("SNR of ΔF/F (99th percentile above the median, in noise s.d.)")
    ax.set_ylabel("ROIs")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"{sid}: signal-to-noise ratio", fontsize=10)
    fig.tight_layout()
    p = out / f"{sid}_snr.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return {"path": p.name, "caption": (
        f"Signal-to-noise ratio of each ROI in {sid}: how far its large calcium transients (the 99th "
        "percentile of ΔF/F) rise above its median, in units of the frame-to-frame noise. Pure Gaussian "
        "noise gives about 2.3; cells with clear, frequent transients reach 5-20 or more, and sparsely "
        "firing cells (active in under 1 % of frames) can sit near 3-4 although their transients are real. "
        "Accepted cells (green) should sit mostly to the right of the other "
        "ROIs (grey).")}
