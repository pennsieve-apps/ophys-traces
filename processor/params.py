"""
Parameters from the environment (names as in app.yml), validated. A bad value raises
BadParameter, which main() turns into exit code 2.
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

BASELINES = ("maximin", "percentile")
INDICATORS = ("auto", "gcamp6f", "gcamp6s", "gcamp6", "gcamp8", "other")


class BadParameter(ValueError):
    pass


def _get(name: str, default: str) -> str:
    v = os.environ.get(name)
    return default if v is None or v.strip() == "" else v.strip()


def _num(name: str, default, lo=None, hi=None, integer=False, allow_empty=False):
    raw = _get(name, "" if default is None else str(default))
    if raw == "" and allow_empty:
        return None
    try:
        v = float(raw)
    except ValueError:
        raise BadParameter(f"{name}={raw!r}: expected a number") from None
    if not math.isfinite(v):
        raise BadParameter(f"{name}={raw!r}: expected a finite number")
    if integer:
        if v != int(v):
            raise BadParameter(f"{name}={raw!r}: expected a whole number")
        v = int(v)
    if lo is not None and v < lo:
        raise BadParameter(f"{name}={raw!r}: must be >= {lo:g}")
    if hi is not None and v > hi:
        raise BadParameter(f"{name}={raw!r}: must be <= {hi:g}")
    return v


def _bool(name: str, default: bool) -> bool:
    v = _get(name, "true" if default else "false").lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise BadParameter(f"{name}={v!r}: expected true or false")


@dataclass
class Params:
    neuropil_coefficient: float
    baseline: str
    baseline_window_s: float
    baseline_percentile: float
    tau: float                     # 0 = from the Suite2p run (ops["tau"]), else 0.7
    indicator: str
    cascade_model: str             # "auto" or a bundled model name
    classifier_threshold: float | None   # None = Suite2p's own iscell decision
    example_rois: int
    oasis_threshold_sd: float
    cascade_all_rois: bool
    use_gpu: str
    threads: int
    seed: int

    def as_env(self) -> dict:
        return {k.upper(): ("" if v is None else v) for k, v in asdict(self).items()}


def read_params() -> Params:
    baseline = _get("BASELINE", "maximin").lower()
    if baseline not in BASELINES:
        raise BadParameter(f"BASELINE={baseline!r}: expected one of {', '.join(BASELINES)}")
    indicator = _get("INDICATOR", "auto").lower()
    if indicator not in INDICATORS:
        raise BadParameter(f"INDICATOR={indicator!r}: expected one of {', '.join(INDICATORS)}")
    use_gpu = _get("USE_GPU", "auto").lower()
    use_gpu = {"1": "true", "yes": "true", "cuda": "true", "gpu": "true", "on": "true",
               "0": "false", "no": "false", "cpu": "false", "off": "false"}.get(use_gpu, use_gpu)
    if use_gpu not in ("auto", "true", "false"):
        raise BadParameter(f"USE_GPU={use_gpu!r}: expected auto, true or false")
    return Params(
        neuropil_coefficient=_num("NEUROPIL_COEFFICIENT", 0.7, lo=0, hi=1.5),
        baseline=baseline,
        baseline_window_s=_num("BASELINE_WINDOW_S", 60, lo=1, hi=3600),
        baseline_percentile=_num("BASELINE_PERCENTILE", 8, lo=0, hi=50),
        tau=_num("TAU", 0, lo=0, hi=10),
        indicator=indicator,
        cascade_model=_get("CASCADE_MODEL", "auto"),
        classifier_threshold=_num("CLASSIFIER_THRESHOLD", None, lo=0, hi=1, allow_empty=True),
        example_rois=_num("EXAMPLE_ROIS", 8, lo=0, hi=50, integer=True),
        oasis_threshold_sd=_num("OASIS_THRESHOLD_SD", 3, lo=0, hi=50),
        cascade_all_rois=_bool("CASCADE_ALL_ROIS", False),
        use_gpu=use_gpu,
        threads=_num("THREADS", 0, lo=0, integer=True),
        seed=_num("SEED", 42, lo=0, integer=True),
    )
