"""
ophys-traces: neuropil correction, ΔF/F, CASCADE spike rates and OASIS events from Suite2p output,
written as one NWB ophys file per session plus rois.csv and figures.

INPUT_DIR: Suite2p folders (the suite2p app's output, or any lab's `suite2p/plane<k>/`), optional
session NWB files (identity), optional sessions.csv. OUTPUT_DIR/ophys-traces/: see README.md.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from processor.params import BadParameter, read_params
from processor.resources import cgroup_memory_peak_mb, choose_device, resolve_threads

APP = "ophys-traces"
CITE_CASCADE = "10.1038/s41593-021-00895-5"
CITE_CASCADE_GC8 = "10.1038/s41592-026-03183-x"
CITE_OASIS = "10.1371/journal.pcbi.1005423"
CITE_SUITE2P = "10.64898/2026.02.04.703741"
DEFAULT_TAU = 0.7
log = logging.getLogger("processor")


def _version() -> str:
    for p in (Path(__file__).resolve().parent.parent / "VERSION", Path("/processor/VERSION")):
        if p.is_file():
            return p.read_text().strip()
    return "0.0.0"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ru = ru / 1024 if sys.platform != "darwin" else ru / 2**20
    return round(max(ru, cgroup_memory_peak_mb() or 0), 1)


def _tool_versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for pkg in ("numpy", "scipy", "torch", "pynwb", "hdmf", "h5py", "oasis-deconv"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    out["CascadeTorch"] = os.environ.get("CASCADETORCH_COMMIT", "unknown")
    return out


def process_session(s, p, out: Path, device: str) -> dict:
    import numpy as np
    from processor import cascade_infer, figures, nwb_out, traces
    from processor.inputs import load_plane

    sid = s.session_id
    warnings: list[str] = []
    identity, id_source = {}, "none"
    if s.nwb is not None:
        try:
            identity, id_source = nwb_out.read_identity(s.nwb), s.nwb.name
        except OSError as exc:
            warnings.append(f"session NWB {s.nwb.name} unreadable: {exc}")
    if not identity and s.session_json.get("source_nwb"):
        identity, id_source = dict(s.session_json["source_nwb"]), "suite2p session.json"

    loaded = [load_plane(pl) for pl in s.planes]
    fs0 = float(loaded[0]["ops"].get("fs") or s.session_json.get("frame_rate") or 0)
    tau = p.tau or float(loaded[0]["ops"].get("tau") or 0) or DEFAULT_TAU
    family, fam_why = cascade_infer.family_for(identity.get("indicator"), p.indicator)

    planes, rows = [], []
    t_session = time.time()
    # pass 1: ΔF/F, noise and SNR per plane
    for pl, d in zip(s.planes, loaded):
        times = d["times"]
        fs = float(d["ops"].get("fs") or fs0)
        if len(times) > 1:
            fs_ts = (len(times) - 1) / (times[-1] - times[0])
            if abs(fs_ts - fs) / fs > 0.001:
                warnings.append(f"plane{pl.index}: timestamps give {fs_ts:.4f} Hz, ops['fs'] {fs:.4f} Hz; "
                                "using the timestamps' rate")
                fs = fs_ts
        n = len(d["stat"])
        iscell = d["iscell"]
        if iscell is None or len(iscell) != n:
            warnings.append(f"plane{pl.index}: no iscell.npy; every ROI is accepted")
            prob = np.full(n, np.nan)
            acc = np.ones(n, bool)
        else:
            prob = np.asarray(iscell[:, 1], float)
            acc = prob >= p.classifier_threshold if p.classifier_threshold is not None else iscell[:, 0] > 0
        dff, F0, floored = traces.dff(d["F"], d["Fneu"], fs, p.neuropil_coefficient, p.baseline,
                                      p.baseline_window_s, p.baseline_percentile)
        planes.append({"plane": pl.index, "stat": d["stat"], "F": np.asarray(d["F"]), "Fneu": np.asarray(d["Fneu"]),
                       "dff": dff, "F0": F0, "f0_floored": floored, "times": times, "fs": fs,
                       "iscell_prob": prob, "accepted": acc, "roi_ids": np.arange(n), "ops": d["ops"],
                       "noise": traces.noise_level(dff, fs), "snr": traces.snr(dff),
                       "time_source": d["time_source"]})
    import numpy as np
    noise_acc = np.concatenate([pl["noise"][pl["accepted"]] for pl in planes]) if planes else np.array([])
    noise_all = np.concatenate([pl["noise"] for pl in planes])
    ref = noise_acc if noise_acc.size else noise_all
    p90 = float(np.nanpercentile(ref, 90)) if ref.size and np.isfinite(ref).any() else 5.0
    models = cascade_infer.available_models()
    model, why = cascade_infer.choose_model(models, family, fs0 or planes[0]["fs"], p90, p.cascade_model)
    log.info(f"{sid}: CASCADE model {model['name']} ({fam_why}; {why}); OASIS tau {tau:g} s")

    # pass 2: spike inference, events
    for pl in planes:
        tc = time.time()
        n = pl["dff"].shape[0]
        sel = np.arange(n) if p.cascade_all_rois else np.where(pl["accepted"])[0]
        pl["cascade"] = np.full(pl["dff"].shape, np.nan, np.float32)
        pl["cascade_edge"], pl["resampled"] = 0, False
        if len(sel):
            with contextlib.redirect_stdout(io.StringIO()):
                res = cascade_infer.infer(pl["dff"][sel], pl["times"], pl["fs"], model, device=device,
                                          progress=lambda b, n_: log.info(f"{sid}: CASCADE {b}/{n_} ROIs"))
            pl["cascade"][sel] = res["rate"]
            pl["cascade_edge"], pl["resampled"] = res["edge_frames"], res["resampled"]
        t_cascade = time.time() - tc
        to = time.time()
        pl["oasis"] = traces.oasis_events(pl["dff"], pl["fs"], tau, p.oasis_threshold_sd)
        t_oasis = time.time() - to
        pl["cascade_model"] = model["name"]
        pl["seconds"] = {"cascade": round(t_cascade, 1), "oasis": round(t_oasis, 1)}
        log.info(f"{sid} plane{pl['plane']}: {n} ROIs ({int(pl['accepted'].sum())} accepted), CASCADE on {len(sel)} "
                 f"{t_cascade:.1f} s, OASIS {t_oasis:.1f} s")
        dur = pl["times"][-1] - pl["times"][0] + 1 / pl["fs"]
        valid = slice(pl["cascade_edge"], max(pl["cascade_edge"] + 1, len(pl["times"]) - pl["cascade_edge"]))
        for i, st in enumerate(pl["stat"]):
            med = st.get("med", [np.nan, np.nan])
            rows.append({
                "session_id": sid, "plane": pl["plane"], "roi_id": i, "accepted": bool(pl["accepted"][i]),
                "iscell_prob": round(float(pl["iscell_prob"][i]), 4) if np.isfinite(pl["iscell_prob"][i]) else "",
                "x": float(med[1]), "y": float(med[0]), "npix": int(st.get("npix", len(st["ypix"]))),
                "radius": round(float(st.get("radius", np.nan)), 2),
                "compact": round(float(st.get("compact", np.nan)), 3),
                "snr": round(float(pl["snr"][i]), 3), "noise_level": round(float(pl["noise"][i]), 3),
                "f0_median": round(float(np.median(pl["F0"][i])), 2),
                "f0_floored": round(float(pl["f0_floored"][i]), 3),
                "dff_max": round(float(np.nanmax(pl["dff"][i])), 4),
                "cascade_rate_hz": (round(float(np.mean(pl["cascade"][i, valid])), 4)
                                    if np.isfinite(pl["cascade"][i, valid]).all() else ""),
                "oasis_events": int((pl["oasis"][i] > 0).sum()),
                "oasis_event_rate_hz": round(float((pl["oasis"][i] > 0).sum() / dur), 4),
                **{k: v for k, v in s.sheet.items() if k in ("subject_id", "condition", "group")},
            })

    # outputs
    meta = {"neuropil_coefficient": p.neuropil_coefficient, "baseline": f"{p.baseline} {p.baseline_window_s:g} s",
            "tau": tau, "oasis_threshold_sd": p.oasis_threshold_sd, "version": _version(),
            "source_file": s.nwb.name if s.nwb is not None and id_source != "none" else
            (s.session_json.get("companion_nwb") or (s.session_json.get("source") or [None])[0])}
    nwb_path = out / f"{sid}_ophys.nwb"
    warnings += nwb_out.write(nwb_path, sid, identity, planes, meta)
    figs = []
    import numpy as np
    for pl in planes:
        tag = sid if len(planes) == 1 else f"{sid}_plane{pl['plane']}"
        labels = [f"ROI {i}" if len(planes) == 1 else f"p{pl['plane']} ROI {i}" for i in range(len(pl["stat"]))]
        f = figures.example_traces(tag, out, pl["times"], pl["dff"], pl["cascade"], pl["oasis"], pl["snr"],
                                   pl["accepted"], labels, p.example_rois, model["name"])
        if f:
            figs.append(f)
        figs.append(figures.noise_hist(tag, out, pl["noise"], pl["accepted"], model))
        figs.append(figures.snr_hist(tag, out, pl["snr"], pl["accepted"]))
    acc_all = np.concatenate([pl["accepted"] for pl in planes])
    snr_all = np.concatenate([pl["snr"] for pl in planes])
    rate_all = np.array([np.nan if r["cascade_rate_hz"] == "" else r["cascade_rate_hz"] for r in rows], float)
    ev_all = np.array([r["oasis_event_rate_hz"] for r in rows])
    t0, t1 = planes[0]["times"][0], planes[0]["times"][-1]
    summary = {
        "n_rois": int(len(acc_all)), "n_accepted": int(acc_all.sum()),
        "n_planes": len(planes), "n_frames": int(len(planes[0]["times"])),
        "frame_rate": round(planes[0]["fs"], 4), "t_start": round(float(t0), 4), "t_stop": round(float(t1), 4),
        "time_source": planes[0]["time_source"], "identity_source": id_source,
        "indicator": identity.get("indicator"), "cascade_model": model["name"], "cascade_model_reason": f"{fam_why}; {why}",
        "cascade_resampled": any(pl["resampled"] for pl in planes), "cascade_edge_frames": planes[0]["cascade_edge"],
        "median_snr_accepted": round(float(np.nanmedian(snr_all[acc_all])), 3) if acc_all.any() else None,
        "median_noise_level_accepted": round(float(np.nanmedian(noise_acc)), 3) if noise_acc.size else None,
        "mean_cascade_rate_hz_accepted": round(float(np.nanmean(rate_all[acc_all])), 4) if acc_all.any() else None,
        "mean_oasis_event_rate_hz_accepted": round(float(np.nanmean(ev_all[acc_all])), 4) if acc_all.any() else None,
        "n_f0_floored": int(sum(int((pl["f0_floored"] > 0.1).sum()) for pl in planes)),
        "tau": tau, "cascade_rois": "all" if p.cascade_all_rois else "accepted", "seconds": round(time.time() - t_session, 1),
        **{k: v for k, v in s.sheet.items() if k in ("subject_id", "condition", "group")},
    }
    return {"summary": summary, "rows": rows, "figures": figs, "warnings": warnings,
            "cascade": {"model": model["name"], "family": family, "indicator": identity.get("indicator"),
                        "indicator_parameter": p.indicator, "reason": f"{fam_why}; {why}"},
            "inputs": [str(pl.path) for pl in s.planes] + ([str(s.nwb)] if s.nwb else []),
            "family": family}


def main() -> int:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)
    log.handlers = [h]
    log.setLevel(logging.INFO)
    log.propagate = False
    started, t0 = _now(), time.time()
    version = _version()
    try:
        p = read_params()
    except BadParameter as exc:
        log.error(f"bad parameter: {exc}")
        return 2
    in_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    n_threads, src = resolve_threads(p.threads)
    log.info(f"{APP} {version}: INPUT_DIR={in_dir} OUTPUT_DIR={out_dir}, {n_threads} threads ({src})")
    import numpy as np
    import torch
    torch.set_num_threads(n_threads)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)
    dev = choose_device(p.use_gpu)
    log.info(dev["reason"])
    device = "cuda" if dev["accelerator"] == "gpu" else "cpu"

    from processor import cascade_infer
    from processor.inputs import discover
    if p.cascade_model != "auto" and p.cascade_model not in {m["name"] for m in cascade_infer.available_models()}:
        log.error(f"CASCADE_MODEL={p.cascade_model!r} is not bundled (see README for the list)")
        return 2
    if not in_dir.is_dir():
        log.error(f"INPUT_DIR {in_dir} does not exist")
        return 1
    sessions, warnings = discover(in_dir)
    for w in warnings:
        log.warning(w)
    if not sessions:
        log.error("no Suite2p output found (folders with stat.npy, F.npy, Fneu.npy, ops.npy)")
        return 1
    log.info(f"{len(sessions)} session(s): " + ", ".join(
        f"{s.session_id} ({len(s.planes)} plane(s){', NWB ' + s.nwb.name if s.nwb else ''})" for s in sessions))

    top = out_dir / APP
    top.mkdir(parents=True, exist_ok=True)
    results, statuses, figs, rows, inputs, cascade_choice = {}, {}, [], [], [], {}
    families = set()
    for s in sessions:
        ts = time.time()
        try:
            r = process_session(s, p, top, device)
            results[s.session_id] = r["summary"]
            statuses[s.session_id] = {"status": "ok", "seconds": round(time.time() - ts, 1)}
            figs += r["figures"]
            rows += r["rows"]
            inputs += r["inputs"]
            families.add(r["family"])
            cascade_choice[s.session_id] = r["cascade"]
            for w in r["warnings"]:
                warnings.append(f"{s.session_id}: {w}")
            sm = r["summary"]
            log.info(f"{s.session_id}: ok, {sm['n_rois']} ROIs ({sm['n_accepted']} accepted), {sm['n_frames']} frames, "
                     f"median SNR {sm['median_snr_accepted']}, mean CASCADE rate {sm['mean_cascade_rate_hz_accepted']} Hz, "
                     f"{time.time() - ts:.1f} s")
        except Exception as exc:
            log.error(f"{s.session_id}: failed: {exc}")
            traceback.print_exc(file=sys.stdout)
            statuses[s.session_id] = {"status": "failed", "seconds": round(time.time() - ts, 1), "error": str(exc)}
            warnings.append(f"{s.session_id}: failed: {exc}")
            results[s.session_id] = {"status": "failed", "error": str(exc)}
    for sid, r in results.items():
        r.setdefault("status", statuses[sid]["status"])

    if rows:
        cols = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(top / "rois.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    summary = {"app": APP, "version": version, "sessions": results, "figures": figs,
               "tables": [{"path": "rois.csv", "caption": "One row per ROI and session: accepted, classifier "
                           "probability, position, size, SNR, noise level, baseline, mean CASCADE rate, OASIS events."}],
               "warnings": warnings}
    (top / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    gpu = None
    if device == "cuda":
        gpu = {"name": torch.cuda.get_device_name(0), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
    cites = [CITE_CASCADE] + ([CITE_CASCADE_GC8] if "GC8" in families else []) + [CITE_OASIS, CITE_SUITE2P]
    variant = os.environ.get("APP_VARIANT") or ("gpu" if os.environ.get("TORCH_FLAVOUR") == "cuda" else "cpu")
    prov = {"app": APP, "version": version, "variant": variant, "tools": _tool_versions(),
            "parameters": p.as_env(),
            "inputs": sorted({os.path.relpath(i, in_dir) for i in inputs}),
            "started": started, "finished": _now(), "wall_seconds": round(time.time() - t0, 1),
            "peak_rss_mb": _peak_rss_mb(), "gpu": gpu, "device": dev["reason"],
            "sessions": statuses, "cascade": cascade_choice, "citations": cites}
    (top / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    n_ok = sum(1 for v in statuses.values() if v["status"] == "ok")
    log.info(f"done: {n_ok}/{len(sessions)} session(s) ok in {time.time() - t0:.1f} s, peak memory {prov['peak_rss_mb']} MB")
    return 0 if n_ok else 1
