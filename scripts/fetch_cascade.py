"""
Build-time only (Dockerfile): fetch CascadeTorch's source at a pinned commit and the pretrained
CASCADE models listed in cascade-models.txt, verify every file's sha256, apply the upstream
spike-timing fix, and check that each model's config names the model it is filed under.

  python scripts/fetch_cascade.py /opt/cascade cascade-models.txt

CascadeTorch (PTRRupprecht/CascadeTorch, GPL-3.0) is not on PyPI and its setup.py installs no
code (cascade2p/ has no __init__.py; upstream issue #3), so the package folder is placed on
PYTHONPATH instead. Its last commit (c6978d5, 2026-08-09) predates the CASCADE fix of 2026-10-27
(HelmchenLabSoftware/Cascade issue #90, commit bcc00141a6a8: predictions were one frame late);
CascadeTorch has the same two lines, so the same fix is applied here (see
patches/cascadetorch-issue90-off-by-one.patch). The model archives come from the links in
CascadeTorch's Pretrained_models/available_models_CascadeTorch.yaml (drive.switch.ch).
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

COMMIT = "c6978d5ff33edad8792c76e040412f0636913092"
RAW = f"https://raw.githubusercontent.com/PTRRupprecht/CascadeTorch/{COMMIT}/"
SOURCES = {
    "cascade2p/cascade.py": "7adbcfb7c7334340005f5bc63f2f20ef1b2e4f760d40573fefdf4db0a6b288a0",
    "cascade2p/config.py": "d1e6158f87541f7ced212980fcd500f0be6556f40393a46d674f19b74baf7d03",
    "cascade2p/utils.py": "7b14125af39dca317f5ccd0e965ee730673c614305d553fc0d6878c9903a45b5",
    "cascade2p/utils_discrete_spikes.py": "78594e1f50beb6884352e11821c9797c124e83a72e1ae6c35a2e898dce98799f",
    "cascade2p/checks.py": "2a7561d1c76bf61b4613fed5bba730615d066b94d1faeefbdfee13f52df75c06",
    "LICENSE": "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986",
}
# upstream HelmchenLabSoftware/Cascade commit bcc00141a6a8 (issue #90), applied to the same lines
FIX = {
    "cascade2p/utils.py": ("start = int(before_frac * window_size -1)", "start = int(before_frac * window_size)"),
    "cascade2p/utils_discrete_spikes.py": ("offset = np.argmax(nnan_indices==True) - 1",
                                           "offset = np.argmax(nnan_indices==True)"),
}


def fetch(url: str, tries: int = 4) -> bytes:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "pennsieve-app-build"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception as exc:  # retried: the model host is occasionally slow
            if i == tries - 1:
                raise
            print(f"retry {url}: {exc}", flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("unreachable")


def check(data: bytes, sha: str, what: str) -> None:
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"sha256 mismatch for {what}: expected {sha}, got {got}")


def main(dest: Path, models_list: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for rel, sha in SOURCES.items():
        data = fetch(RAW + rel)
        check(data, sha, rel)
        text = data.decode()
        if rel in FIX:
            old, new = FIX[rel]
            if text.count(old) != 1:
                raise SystemExit(f"{rel}: the line to fix was not found exactly once")
            text = text.replace(old, new)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    (dest / "cascade2p" / "__init__.py").write_text("")
    (dest / "COMMIT").write_text(COMMIT + " + HelmchenLabSoftware/Cascade bcc00141a6a8 (issue #90)\n")
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    models = dest / "Pretrained_models"
    for line in models_list.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, url, sha = line.split()
        data = fetch(url)
        check(data, sha, name)
        target = models / name
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(target)
        cfg = yaml.load((target / "config.yaml").read_text())
        if cfg.get("model_name") != name:
            raise SystemExit(f"{name}: its config.yaml names {cfg.get('model_name')!r}")
        n = len(list(target.glob("*.pth")))
        print(f"{name}: {n} weight files, {cfg['sampling_rate']} Hz, noise levels "
              f"{min(cfg['noise_levels'])}-{max(cfg['noise_levels'])}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
