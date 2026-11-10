"""
The session's NWB ophys file (conventions.md, "Calcium imaging", port `traces`):

processing/ophys/
  ImageSegmentation/PlaneSegmentation        pixel_mask (x = row, y = column, weight), iscell (classifier
                                             probability), accepted (bool), plus roi_id, plane,
                                             snr, noise_level, f0_floored
  Fluorescence/RoiResponseSeries             F (raw ROI fluorescence, Suite2p units)
  Fluorescence/Neuropil                      F_neu
  DfOverF/RoiResponseSeries                  ΔF/F (fraction)
  Deconvolved/RoiResponseSeries              CASCADE spike rate (spikes/s; NaN for ROIs not inferred)
  Deconvolved/OASIS                          OASIS event amplitudes (ΔF/F units)
  SummaryImages/{mean, max_projection, correlation}   from Suite2p's ops
all on the session clock (seconds): regular series (every interval within 0.1 % of the median, no
gaps) as `starting_time` + `rate`, irregular ones with explicit `timestamps` (one array, linked by
the other series). With more
than one plane, plane k > 0 gets `PlaneSegmentation_plane<k>`, `RoiResponseSeries_plane<k>`, ...

The file gets a new UUID identifier; the source NWB's file name and identifier go into `notes`.
Its identity (session_start_time, session_id, session_description, subject, keywords,
timestamps_reference_time, ...) is copied from the session NWB, else from the suite2p app's
session.json (`source_nwb`), so files from different apps of the same session line up.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

UNKNOWN_START = datetime(1970, 1, 1, tzinfo=timezone.utc)


def read_identity(nwb_path: Path) -> dict:
    """Identity, subject and imaging plane details of a session NWB (read with h5py: robust to
    extensions pynwb does not have)."""
    import h5py

    def s(v):
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray):
            return [s(x) for x in v.tolist()] if v.ndim else s(v[()])
        return v

    out = {}
    with h5py.File(nwb_path, "r") as f:
        for k in ("session_start_time", "identifier", "session_description", "timestamps_reference_time"):
            if k in f:
                out[k] = s(f[k][()])
        g = f.get("general")
        if g is not None:
            for k in ("session_id", "lab", "institution", "experimenter", "experiment_description",
                      "protocol", "keywords"):
                if k in g and hasattr(g[k], "shape"):
                    out[k] = s(g[k][()])
            if "subject" in g:
                out["subject"] = {k: s(g["subject"][k][()]) for k in g["subject"]
                                  if hasattr(g["subject"][k], "shape") and g["subject"][k].shape == ()}
            op = g.get("optophysiology")
            if op is not None:
                for name in op:
                    pl = op[name]
                    if hasattr(pl, "keys") and "indicator" in pl:
                        out["indicator"] = s(pl["indicator"][()])
                        for k in ("location", "description", "excitation_lambda"):
                            if k in pl:
                                out[f"plane_{k}"] = s(pl[k][()])
                        break
    return out


def _dt(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        d = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def regular_rate(times, tol: float = 0.001) -> float | None:
    """The rate when every frame interval is within tol (0.1 %) of the median (no gaps), else None."""
    t = np.asarray(times, np.float64)
    if len(t) < 2:
        return None
    dt = np.diff(t)
    med = float(np.median(dt))
    if med <= 0 or np.any(np.abs(dt - med) > tol * med):
        return None
    return (len(t) - 1) / (t[-1] - t[0])


def write(path: Path, session_id: str, identity: dict, planes: list[dict], meta: dict) -> list[str]:
    """Write the NWB; returns warnings. `planes`: dicts with stat, F, Fneu, dff, cascade, oasis,
    times, iscell_prob, accepted, roi_ids, snr, noise, f0_floored, ops, Ly, Lx, fs."""
    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.base import Images
    from pynwb.file import Subject
    from pynwb.image import GrayscaleImage
    from pynwb.ophys import DfOverF, Fluorescence, ImageSegmentation, OpticalChannel, RoiResponseSeries

    warnings = []
    start = _dt(identity.get("session_start_time"))
    if start is None:
        start = UNKNOWN_START
        warnings.append("session_start_time unknown (no session NWB): set to 1970-01-01; link the "
                        "session's NWB into the `session` input to fix")
    kwargs = dict(
        session_description=identity.get("session_description") or f"two-photon calcium imaging, session {session_id}",
        identifier=str(uuid.uuid4()),      # a new identifier per file (NWB/DANDI: unique)
        session_start_time=start,
        session_id=identity.get("session_id") or session_id,
    )
    src = meta.get("source_file")
    kwargs["notes"] = (f"derived from {src} (identifier {identity.get('identifier', 'unknown')}) "
                       if src else "derived from Suite2p output ") + f"by ophys-traces {meta.get('version', '')}".strip()
    if identity.get("keywords"):
        kw = identity["keywords"]
        kwargs["keywords"] = list(kw) if isinstance(kw, (list, tuple)) else [kw]
    trt = _dt(identity.get("timestamps_reference_time"))
    if trt is not None:
        kwargs["timestamps_reference_time"] = trt
    for k in ("lab", "institution", "experiment_description", "protocol"):
        if isinstance(identity.get(k), str) and identity[k]:
            kwargs[k] = identity[k]
    if identity.get("experimenter"):
        e = identity["experimenter"]
        kwargs["experimenter"] = e if isinstance(e, list) else [e]
    nwb = NWBFile(**kwargs)
    subj = identity.get("subject") or {}
    if subj:
        allowed = ("subject_id", "species", "sex", "age", "genotype", "description", "strain", "weight")
        sk = {k: str(v) for k, v in subj.items() if k in allowed and v not in (None, "")}
        if "date_of_birth" in subj and _dt(subj["date_of_birth"]):
            sk["date_of_birth"] = _dt(subj["date_of_birth"])
        if sk:
            nwb.subject = Subject(**sk)

    device = nwb.create_device(name="Microscope", description="two-photon microscope")
    ophys = nwb.create_processing_module("ophys", "Optical physiology: Suite2p ROIs (suite2p app), "
                                         "neuropil-corrected ΔF/F, CASCADE spike rates and OASIS events "
                                         "(ophys-traces app)")
    seg = ImageSegmentation(name="ImageSegmentation")
    ophys.add(seg)
    fl = Fluorescence(name="Fluorescence")
    dfo = DfOverF(name="DfOverF")
    dec = Fluorescence(name="Deconvolved")
    images = Images(name="SummaryImages", description="Suite2p summary images per plane")
    ophys.add(fl)
    ophys.add(dfo)
    ophys.add(dec)
    ts_link = None
    for pl in planes:
        k = pl["plane"]
        sfx = "" if k == 0 else f"_plane{k}"
        ch = OpticalChannel(name="OpticalChannel", description="functional channel",
                            emission_lambda=float("nan"))
        ip = nwb.create_imaging_plane(
            name=f"ImagingPlane{sfx}", optical_channel=ch, description=f"plane {k}",
            device=device, excitation_lambda=float("nan"), imaging_rate=float(pl["fs"]),
            indicator=str(identity.get("indicator") or "unknown"),
            location=str(identity.get("plane_location") or identity.get("location") or "unknown"))
        ps = seg.create_plane_segmentation(
            name=f"PlaneSegmentation{sfx}", imaging_plane=ip,
            description="Suite2p ROIs (all of them; `accepted` marks those analysed downstream)")
        ps.add_column("roi_id", "ROI index in Suite2p's stat.npy of this plane")
        ps.add_column("plane", "imaging plane")
        ps.add_column("iscell", "Suite2p classifier probability that the ROI is a cell")
        ps.add_column("accepted", "True when the ROI is analysed downstream (classified as a cell)")
        ps.add_column("snr", "signal-to-noise ratio of ΔF/F (see README)")
        ps.add_column("noise_level", "CASCADE noise level of ΔF/F (% / sqrt(Hz))")
        ps.add_column("f0_floored", "fraction of frames where the ΔF/F baseline hit its floor")
        for i, s in enumerate(pl["stat"]):
            lam = np.asarray(s["lam"], float)
            # NWB pixel_mask (x, y, weight) with x = the first (row) axis of the image, as Suite2p's own
            # NWB export and the Allen files write it (conventions.md, traces): x = ypix, y = xpix
            mask = [(int(r), int(c), float(w)) for r, c, w in zip(s["ypix"], s["xpix"], lam)]
            ps.add_roi(pixel_mask=mask, roi_id=int(pl["roi_ids"][i]), plane=int(k),
                       iscell=float(pl["iscell_prob"][i]), accepted=bool(pl["accepted"][i]),
                       snr=float(pl["snr"][i]), noise_level=float(pl["noise"][i]),
                       f0_floored=float(pl["f0_floored"][i]))
        region = ps.create_roi_table_region(region=list(range(len(pl["stat"]))), description="all ROIs")
        times = np.asarray(pl["times"], np.float64)

        regular = regular_rate(times)

        def series(name, data, unit, desc):
            nonlocal ts_link
            d = H5DataIO(np.ascontiguousarray(np.asarray(data, np.float32).T), compression="gzip",
                         compression_opts=4, shuffle=True)
            if regular is not None:     # conventions.md "Time": regular series as starting_time + rate
                return RoiResponseSeries(name=name, data=d, rois=region, unit=unit,
                                         starting_time=float(times[0]), rate=regular, description=desc)
            if ts_link is None or not np.array_equal(np.asarray(ts_link.timestamps), times):
                rs = RoiResponseSeries(name=name, data=d, rois=region, unit=unit, timestamps=times,
                                       description=desc)
                ts_link = rs
            else:
                rs = RoiResponseSeries(name=name, data=d, rois=region, unit=unit, timestamps=ts_link,
                                       description=desc)
            return rs

        fl.add_roi_response_series(series(f"RoiResponseSeries{sfx}", pl["F"], "n.a.",
                                          "raw ROI fluorescence F (Suite2p F.npy)"))
        fl.add_roi_response_series(series(f"Neuropil{sfx}", pl["Fneu"], "n.a.",
                                          "neuropil fluorescence (Suite2p Fneu.npy)"))
        dfo.add_roi_response_series(series(f"RoiResponseSeries{sfx}", pl["dff"], "n.a.",
                                           f"ΔF/F of F - {meta['neuropil_coefficient']} x neuropil, "
                                           f"baseline {meta['baseline']}"))
        dec.add_roi_response_series(series(f"RoiResponseSeries{sfx}", pl["cascade"], "hertz",
                                           f"CASCADE spike rate, model {pl['cascade_model']}; NaN for ROIs it was not run on "
                                           "(not accepted, unless CASCADE_ALL_ROIS=true)"))
        dec.add_roi_response_series(series(f"OASIS{sfx}", pl["oasis"], "n.a.",
                                           f"OASIS events (AR(1), tau {meta['tau']} s, minimal event "
                                           f"{meta['oasis_threshold_sd']} noise s.d.)"))
        ops = pl["ops"]
        for key, label in (("meanImg", "mean"), ("max_proj", "max_projection"), ("Vcorr", "correlation")):
            im = ops.get(key)
            if im is not None and np.ndim(im) == 2:
                images.add_image(GrayscaleImage(name=f"{label}{sfx}", data=np.asarray(im, np.float32),
                                                description=f"Suite2p {key}"))
    if len(images.images):
        ophys.add(images)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return warnings
