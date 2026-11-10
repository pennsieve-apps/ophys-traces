"""
A small synthetic input for ophys-traces, for the unit tests, `make run` and the review run:

  python -m tests.make_fixture data/fixture [--layout linked|flat|both] [--seed 3]

- `suite2p/suite2p/sub-synth01_ses-day1/plane0/`: what the suite2p app writes (stat.npy, F.npy,
  Fneu.npy, iscell.npy, ops.npy, spks.npy, timestamps.npy, plus session.json and timestamps.npy
  one level up): 30 ROIs on a 96 x 96 field, 15 Hz, 1,500 frames starting at 100.0 s on the
  session clock. ROIs 0-23 are cells with Poisson spike trains (0.2-2 Hz) convolved with a
  GCaMP6f-like kernel (one spike = about 0.2 ΔF/F after neuropil correction); ROIs 24-29 are not
  cells (iscell 0: noise and slow drift only). Each ROI's F carries 0.7 x its neuropil trace.
- `session/sub-synth01_ses-day1_behavior+ophys.nwb`: the session NWB (identity: start time,
  identifier, subject, imaging plane with indicator GCaMP6f).
- `suite2p/labdata/mouse7_day3/suite2p/plane0/`: a lab's own Suite2p folder (no session.json, no
  timestamps; ops["fs"] = 30 Hz), 12 cells, 1,800 frames, named by `suite2p/sessions.csv`
  (`session_id` `mouse7_day3`, `condition` `ctrl`).

`truth/<session_id>_truth.npz` holds the spike trains (tests only).
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SESSIONS = {
    "sub-synth01_ses-day1": dict(fs=15.0, n_frames=1500, n_cells=24, n_noise=6, t0=100.0, suite2p_app=True),
    "mouse7_day3": dict(fs=30.0, n_frames=1800, n_cells=12, n_noise=0, t0=0.0, suite2p_app=False),
}
LY = LX = 96


def _kernel(fs, tau=0.7, rise=0.05):
    t = np.arange(int(5 * tau * fs)) / fs
    k = (1 - np.exp(-t / rise)) * np.exp(-t / tau)
    return k / k.max()


def simulate(cfg, seed):
    rng = np.random.default_rng(seed)
    fs, T, nc, nn = cfg["fs"], cfg["n_frames"], cfg["n_cells"], cfg["n_noise"]
    n = nc + nn
    rates = np.r_[rng.uniform(0.2, 2.0, nc), np.zeros(nn)]
    spikes = rng.poisson(rates[:, None] / fs, (n, T)).astype(np.float32)
    k = _kernel(fs)
    ca = np.array([np.convolve(s, k)[:T] for s in spikes])
    t = np.arange(T) / fs
    neu = 200 + 20 * np.sin(2 * np.pi * t / 40)[None] + rng.normal(0, 3, (n, T))
    base = rng.uniform(150, 250, n)[:, None]
    F = base * (1 + 0.2 * ca) + 0.7 * neu + rng.normal(0, 4, (n, T))
    F[nc:] += 15 * np.sin(2 * np.pi * t / 90 + rng.uniform(0, 6, (nn, 1)))
    stat = []
    yy, xx = np.mgrid[:LY, :LX]
    for i in range(n):
        cy, cx = 10 + (i // 6) * 15, 10 + (i % 6) * 15
        m = np.hypot(yy - cy, xx - cx) <= 5
        ypix, xpix = yy[m], xx[m]
        lam = np.exp(-((ypix - cy) ** 2 + (xpix - cx) ** 2) / 20.0).astype(np.float32)
        stat.append({"ypix": ypix, "xpix": xpix, "lam": lam, "med": [cy, cx], "npix": int(m.sum()),
                     "radius": 5.0, "compact": 1.05, "overlap": np.zeros(m.sum(), bool)})
    prob = np.r_[rng.uniform(0.6, 0.99, nc), rng.uniform(0.02, 0.3, nn)]
    iscell = np.c_[(prob > 0.5).astype(float), prob]
    img = rng.uniform(100, 200, (LY, LX)).astype(np.float32)
    for st in stat:                       # cells are bright in the mean image (row = ypix, col = xpix)
        img[st["ypix"], st["xpix"]] += 400
    ops = {"fs": fs, "tau": 0.7, "Ly": LY, "Lx": LX, "meanImg": img, "max_proj": img * 1.5,
           "Vcorr": img / 200, "yrange": [0, LY], "xrange": [0, LX], "nframes": T}
    return {"F": F.astype(np.float32), "Fneu": neu.astype(np.float32), "stat": np.array(stat, dtype=object),
            "iscell": iscell, "ops": ops, "spikes": spikes, "times": cfg["t0"] + np.arange(T) / fs}


def write_plane(d: Path, sim, timestamps: bool):
    d.mkdir(parents=True)
    np.save(d / "stat.npy", sim["stat"], allow_pickle=True)
    np.save(d / "F.npy", sim["F"])
    np.save(d / "Fneu.npy", sim["Fneu"])
    np.save(d / "spks.npy", np.zeros_like(sim["F"]))
    np.save(d / "iscell.npy", sim["iscell"])
    np.save(d / "ops.npy", sim["ops"], allow_pickle=True)
    if timestamps:
        np.save(d / "timestamps.npy", sim["times"])


def write_nwb(path: Path, sid: str):
    try:
        import pynwb  # noqa: F401
    except ImportError:   # the review generator image has no pynwb: no session NWB then
        return
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.file import Subject
    from pynwb.ophys import OpticalChannel
    nwb = NWBFile(session_description="synthetic session", identifier=f"synthetic-{sid}",
                  session_start_time=datetime(2026, 1, 15, 10, 30, tzinfo=timezone.utc),
                  session_id="day1", lab="synthetic lab", institution="Pennsieve")
    nwb.subject = Subject(subject_id="synth01", species="Mus musculus", sex="M", age="P100D")
    dev = nwb.create_device(name="Microscope")
    ch = OpticalChannel(name="green", description="GCaMP", emission_lambda=510.0)
    nwb.create_imaging_plane(name="ImagingPlane", optical_channel=ch, imaging_rate=15.0, description="p",
                             device=dev, excitation_lambda=920.0, indicator="GCaMP6f", location="VISp")
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def make(out: Path, layout: str = "both", seed: int = 3):
    import json
    out = Path(out)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    sims = {}
    for i, (sid, cfg) in enumerate(SESSIONS.items()):
        sims[sid] = simulate(cfg, seed + i)
        np.savez_compressed(out / "truth" / f"{sid}_truth.npz", spikes=sims[sid]["spikes"],
                            times=sims[sid]["times"], n_cells=cfg["n_cells"])
    for lay in (["linked", "flat"] if layout == "both" else [layout]):
        root = out / lay
        if root.exists():
            shutil.rmtree(root)
        s2p = root / "suite2p" if lay == "linked" else root
        ses = root / "session" if lay == "linked" else root
        ses.mkdir(parents=True, exist_ok=True)
        sid = "sub-synth01_ses-day1"
        sdir = s2p / "suite2p" / sid
        write_plane(sdir / "plane0", sims[sid], timestamps=True)
        np.save(sdir / "timestamps.npy", sims[sid]["times"])
        (sdir / "session.json").write_text(json.dumps({"session_id": sid, "frame_rate": 15.0, "n_frames": 1500}))
        write_nwb(ses / f"{sid}_behavior+ophys.nwb", sid)
        write_plane(s2p / "labdata" / "mouse7_day3" / "suite2p" / "plane0", sims["mouse7_day3"], timestamps=False)
        (s2p / "sessions.csv").write_text("session_id,path,subject_id,condition\nmouse7_day3,mouse7_day3,m7,ctrl\n")
    print(f"fixture written to {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--layout", choices=["linked", "flat", "both"], default="both")
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    make(Path(a.out), a.layout, a.seed)


if __name__ == "__main__":
    main()
