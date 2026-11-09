"""
The synthetic fixture (tests/make_fixture.py) through the processor, without Docker. Needs
CascadeTorch and the CASCADE models: CASCADE_MODELS (default /opt/cascade/Pretrained_models) and
cascade2p on PYTHONPATH (see the Makefile's `test`).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from tests.make_fixture import make

SID, LAB = "sub-synth01_ses-day1", "mouse7_day3"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    make(root, "both")
    return root


def _run(mp, in_dir, out_dir, **env):
    from processor.main import main
    for k in ("NEUROPIL_COEFFICIENT", "BASELINE", "TAU", "INDICATOR", "CASCADE_MODEL", "CLASSIFIER_THRESHOLD",
              "CASCADE_ALL_ROIS", "OASIS_THRESHOLD_SD"):
        mp.delenv(k, raising=False)
    for k, v in {"INPUT_DIR": in_dir, "OUTPUT_DIR": out_dir, "USE_GPU": "false", "THREADS": 2, **env}.items():
        mp.setenv(k, str(v))
    return main()


@pytest.fixture(scope="session")
def run(fixture_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        code = _run(mp, fixture_dir / "linked", out)
    finally:
        mp.undo()
    return code, out / "ophys-traces"


def test_outputs(run):
    code, top = run
    assert code == 0
    for f in ("summary.json", "provenance.json", "rois.csv", f"{SID}_ophys.nwb", f"{LAB}_ophys.nwb",
              f"{SID}_example_traces.png", f"{SID}_noise_levels.png", f"{SID}_snr.png"):
        assert (top / f).is_file(), f
    s = json.loads((top / "summary.json").read_text())
    assert s["sessions"][SID]["n_rois"] == 30 and s["sessions"][SID]["n_accepted"] == 24
    assert s["sessions"][SID]["identity_source"].endswith(".nwb")
    assert s["sessions"][SID]["cascade_model"] == "Global_EXC_15Hz_smoothing100ms"
    assert s["sessions"][LAB]["cascade_model"].startswith("Global_EXC_30Hz")
    assert s["sessions"][LAB]["condition"] == "ctrl"
    assert all(f["caption"] for f in s["figures"])
    prov = json.loads((top / "provenance.json").read_text())
    assert "10.1038/s41593-021-00895-5" in prov["citations"]


def test_nwb_contract(run):
    _, top = run
    with h5py.File(top / f"{SID}_ophys.nwb", "r") as f:
        assert f["session_start_time"][()].decode().startswith("2026-01-15T10:30:00")
        import uuid
        uuid.UUID(f["identifier"][()].decode())                       # a new UUID per file
        assert f"identifier synthetic-{SID}" in f["general/notes"][()].decode()
        assert f["general/subject/subject_id"][()].decode() == "synth01"
        o = f["processing/ophys"]
        ps = o["ImageSegmentation/PlaneSegmentation"]
        for c in ("pixel_mask", "iscell", "accepted"):
            assert c in ps
        assert ps["accepted"][:].sum() == 24
        for path in ("Fluorescence/RoiResponseSeries", "Fluorescence/Neuropil", "DfOverF/RoiResponseSeries",
                     "Deconvolved/RoiResponseSeries", "Deconvolved/OASIS"):
            assert o[path]["data"].shape == (1500, 30), path
            assert "timestamps" in o[path]
        ts = o["DfOverF/RoiResponseSeries/timestamps"][:]
        assert ts[0] == pytest.approx(100.0) and len(ts) == 1500   # session clock, not re-zeroed
        rate = o["Deconvolved/RoiResponseSeries/data"][:]
        assert np.isnan(rate[:, 24:]).all() and np.isfinite(rate[:, :24]).all()
    with h5py.File(top / f"{LAB}_ophys.nwb", "r") as f:
        ts = f["processing/ophys/DfOverF/RoiResponseSeries/timestamps"][:]
        assert ts[0] == 0.0 and np.median(np.diff(ts)) == pytest.approx(1 / 30)


@pytest.mark.parametrize("sid", [SID, LAB])
def test_spike_inference_matches_planted_spikes(run, fixture_dir, sid):
    _, top = run
    truth = np.load(fixture_dir / "truth" / f"{sid}_truth.npz")
    n = int(truth["n_cells"])
    with h5py.File(top / f"{sid}_ophys.nwb", "r") as f:
        rate = f["processing/ophys/Deconvolved/RoiResponseSeries/data"][:].T[:n]
        ev = f["processing/ophys/Deconvolved/OASIS/data"][:].T[:n]
        fs = 1 / np.median(np.diff(f["processing/ophys/DfOverF/RoiResponseSeries/timestamps"][:]))
    sp = truth["spikes"][:n]
    e = slice(40, -40)
    c_cas = [np.corrcoef(rate[i, e], gaussian_filter1d(sp[i], 1.0)[e])[0, 1] for i in range(n)]
    c_oas = [np.corrcoef(gaussian_filter1d(ev[i], 1.0)[e], gaussian_filter1d(sp[i], 1.0)[e])[0, 1] for i in range(n)]
    assert np.median(c_cas) > 0.6, np.median(c_cas)
    assert np.median(c_oas) > 0.5, np.median(c_oas)
    # CASCADE's calibrated rate is within a factor 2 of the planted rate
    true_rate = sp[:, e].sum() / (n * sp[:, e].shape[1] / fs)
    est = rate[:, e].mean()
    assert 0.5 < est / true_rate < 2.0, (est, true_rate)


def test_rois_csv(run):
    _, top = run
    with open(top / "rois.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 42
    r = [x for x in rows if x["session_id"] == SID]
    assert sum(x["accepted"] == "True" for x in r) == 24
    for col in ("roi_id", "accepted", "iscell_prob", "x", "y", "snr", "noise_level", "cascade_rate_hz",
                "oasis_events"):
        assert col in rows[0]
    snr = np.array([float(x["snr"]) for x in r])
    assert snr[:24].mean() > snr[24:].mean()


def test_flat_layout_and_options(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "o"
    code = _run(monkeypatch, fixture_dir / "flat", out, BASELINE="percentile", CASCADE_ALL_ROIS="true",
                INDICATOR="gcamp8")
    assert code == 0
    s = json.loads((out / "ophys-traces" / "summary.json").read_text())
    assert s["sessions"][SID]["cascade_model"].startswith("GC8_EXC_15Hz")
    assert s["sessions"][SID]["identity_source"].endswith(".nwb")
    with h5py.File(out / "ophys-traces" / f"{SID}_ophys.nwb", "r") as f:
        assert np.isfinite(f["processing/ophys/Deconvolved/RoiResponseSeries/data"][:]).all()
    assert "10.1038/s41592-026-03183-x" in json.loads((out / "ophys-traces" / "provenance.json").read_text())["citations"]


def test_bad_parameters_and_no_input(fixture_dir, tmp_path, monkeypatch):
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "a", BASELINE="median") == 2
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "b", CASCADE_MODEL="nope") == 2
    assert _run(monkeypatch, fixture_dir / "linked", tmp_path / "c", NEUROPIL_COEFFICIENT="-1") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _run(monkeypatch, empty, tmp_path / "d") == 1


def test_baselines():
    from processor.traces import baseline_maximin, baseline_percentile, dff
    rng = np.random.default_rng(0)
    x = 100 + rng.normal(0, 1, (2, 3000))
    x[:, 1000:1010] += 50
    for b in (baseline_maximin(x, 10, 60), baseline_percentile(x, 10, 60, 8)):
        assert np.all(np.abs(b - 100) < 3)
    d, F0, fl = dff(x + 50, np.full_like(x, 50 / 0.7), 10, 0.7, "maximin", 60, 8)
    assert abs(np.median(d)) < 0.02 and fl.max() == 0
