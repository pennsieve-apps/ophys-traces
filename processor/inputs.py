"""
Finding Suite2p output folders and session NWB files under INPUT_DIR (conventions.md, "Sessions").

Inputs arrive as symlinks, in port subfolders (`suite2p/`, `session/`) or merged at the root, so
everything is found recursively, following symlinks, by name and content.

- A Suite2p plane folder is any folder with `stat.npy`, `F.npy`, `Fneu.npy` and `ops.npy` (and
  usually `iscell.npy`); Suite2p's `combined/` folder is skipped. Plane folders are grouped into
  sessions by their parent folder (the parent of `suite2p/` for a lab's own `<recording>/suite2p/
  plane0/` layout).
- The session id comes from the suite2p app's `session.json`, else a `sessions.csv` row whose
  `path` names the folder, else `sub-<a>_ses-<b>` in the folder names, else the folder name.
- Frame times come from the suite2p app's `timestamps.npy` (session clock), else ops["fs"] (from
  0 s, said in summary.json).
- A session NWB (the `session` input, e.g. Allen's processed file) is matched to a session by
  `sub-<a>_ses-<b>`, sessions.csv or, when there is exactly one session and one NWB, by being the
  only ones.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
SUBSES_RE = re.compile(r"sub-([A-Za-z0-9-]+?)_ses-([A-Za-z0-9-]+)")
REQUIRED = ("stat.npy", "F.npy", "Fneu.npy", "ops.npy")


@dataclass
class Plane:
    path: Path
    index: int


@dataclass
class Session:
    session_id: str
    folder: Path
    planes: list[Plane]
    session_json: dict = field(default_factory=dict)
    sheet: dict = field(default_factory=dict)
    nwb: Path | None = None


def sanitise(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", s).strip("-") or "session"


def subses(names) -> str | None:
    for n in names:
        m = SUBSES_RE.search(n)
        if m:
            ses = m.group(2)
            if UUID_RE.match(ses):
                ses = ses[:8]
            return f"sub-{m.group(1)}_ses-{ses}"
    return None


def walk(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        yield Path(dirpath), dirnames, sorted(f for f in filenames if not f.startswith("."))


def _sheet(root: Path) -> list[dict]:
    rows = []
    for d, _, files in walk(root):
        if "sessions.csv" in files:
            try:
                with open(d / "sessions.csv", newline="", encoding="utf-8-sig") as f:
                    for r in csv.DictReader(f):
                        if r.get("session_id") and r.get("path") is not None:
                            rows.append({k: (v or "").strip() for k, v in r.items() if k})
            except (OSError, csv.Error):
                pass
    return rows


def _match_row(rows, path: Path):
    names = {path.name} | {p.name for p in path.parents}
    for r in rows:
        rp = r["path"].replace("\\", "/").rstrip("/")
        if rp and (Path(rp).name in names or Path(rp).stem in names or str(path).endswith(rp)):
            return r
    for r in rows:
        if r["session_id"] in names:
            return r
    return None


def _plane_index(p: Path, fallback: int) -> int:
    m = re.match(r"plane(\d+)$", p.name)
    return int(m.group(1)) if m else fallback


def discover(root: Path) -> tuple[list[Session], list[str]]:
    warnings: list[str] = []
    rows = _sheet(root)
    plane_dirs, nwbs = [], []
    for d, dirnames, files in walk(root):
        if "combined" in dirnames:
            dirnames.remove("combined")
        if all(f in files for f in REQUIRED):
            plane_dirs.append(d)
        nwbs += [d / f for f in files if f.lower().endswith(".nwb")]
    groups: dict[Path, list[Path]] = {}
    for p in plane_dirs:
        parent = p.parent
        groups.setdefault(parent, []).append(p)
    sessions = []
    for folder, planes in sorted(groups.items()):
        sj = {}
        if (folder / "session.json").is_file():
            try:
                sj = json.loads((folder / "session.json").read_text())
            except (OSError, ValueError):
                sj = {}
        named = folder.parent if folder.name == "suite2p" else folder
        row = _match_row(rows, named)
        if sj.get("session_id"):
            sid = sanitise(sj["session_id"])
        elif row:
            sid = sanitise(row["session_id"])
        else:
            sid = subses([named.name] + [q.name for q in named.parents][:4]) or sanitise(named.name)
        if row is None:
            row = next((r for r in rows if sanitise(r["session_id"]) == sid), None)
        sheet = {k: v for k, v in (row or {}).items() if k not in ("session_id", "path")}
        if not sheet and sj.get("sheet"):
            sheet = dict(sj["sheet"])
        planes = sorted((Plane(p, _plane_index(p, i)) for i, p in enumerate(sorted(planes))), key=lambda q: q.index)
        sessions.append(Session(sid, folder, planes, sj, sheet))
    # duplicates
    ids = [s.session_id for s in sessions]
    for s in sessions:
        if ids.count(s.session_id) > 1:
            new = f"{s.session_id}_{sanitise(s.folder.name)}"
            warnings.append(f"two Suite2p folders give session_id {s.session_id}; this one is {new}")
            s.session_id = new
    # session NWBs
    by_id = {}
    for n in nwbs:
        row = _match_row(rows, n)
        sid = sanitise(row["session_id"]) if row else subses([n.name] + [q.name for q in n.parents][:3])
        if sid:
            by_id.setdefault(sid, n)
    for s in sessions:
        if s.session_id in by_id:
            s.nwb = by_id[s.session_id]
    if len(sessions) == 1 and sessions[0].nwb is None and len(nwbs) == 1:
        sessions[0].nwb = nwbs[0]
        warnings.append(f"{sessions[0].session_id}: using {nwbs[0].name} as the session NWB (the only one; "
                        "its name does not give the same session id)")
    for n in nwbs:
        if not any(s.nwb == n for s in sessions):
            warnings.append(f"session NWB {n.name} matches no Suite2p session; ignored")
    return sessions, warnings


def load_plane(plane: Plane) -> dict:
    p = plane.path
    stat = np.load(p / "stat.npy", allow_pickle=True)
    F = np.load(p / "F.npy", mmap_mode="r")
    Fneu = np.load(p / "Fneu.npy", mmap_mode="r")
    ops = np.load(p / "ops.npy", allow_pickle=True).item()
    iscell = np.load(p / "iscell.npy") if (p / "iscell.npy").is_file() else None
    times, source = None, None
    for cand in (p / "timestamps.npy", p.parent / "timestamps.npy"):
        if cand.is_file():
            t = np.load(cand)
            if len(t) == F.shape[1]:
                times, source = np.asarray(t, float), f"{cand.parent.name}/{cand.name}"
                break
    if times is None:
        fs = float(ops.get("fs", 0) or 0)
        if fs <= 0:
            raise ValueError(f"{p}: no timestamps.npy and no frame rate in ops.npy")
        times, source = np.arange(F.shape[1]) / fs, "frame / ops['fs'] (no timestamps: starts at 0 s)"
    return {"stat": stat, "F": F, "Fneu": Fneu, "ops": ops, "iscell": iscell, "times": times,
            "time_source": source}
