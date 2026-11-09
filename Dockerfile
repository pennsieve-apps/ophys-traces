FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# PyTorch build: cpu or cuda. app.yml's variants set it (cpu variant: CPU-only torch, no NVIDIA libraries;
# gpu variant: the CUDA 12.6 build for the platform's driver 550, which also runs on CPU with USE_GPU=auto).
ARG TORCH_FLAVOUR=cpu

# The complete pinned dependency set (no resolver: see requirements.txt).
COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-deps --prefer-binary \
        -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

# CascadeTorch at a pinned commit plus the upstream CASCADE spike-timing fix (issue #90), and the
# pretrained CASCADE models, every file checked against its sha256 (scripts/fetch_cascade.py).
COPY scripts/fetch_cascade.py cascade-models.txt /build/
RUN python /build/fetch_cascade.py /opt/cascade /build/cascade-models.txt && rm -rf /build

COPY processor/ /processor/processor/
COPY VERSION LICENSE /processor/

# Lambda-safe: at run time only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR (all set by the runtime) are
# writable and nothing the image puts in /tmp or HOME exists. processor/entry.py points matplotlib,
# numba and torch caches at $TMPDIR at startup; everything the app needs at run time is under
# /processor and /opt/cascade (read-only). HOME=/tmp is only the default for ECS.
ENV PYTHONPATH="/processor:/opt/cascade" \
    CASCADE_MODELS=/opt/cascade/Pretrained_models \
    CASCADETORCH_COMMIT=c6978d5ff33edad8792c76e040412f0636913092+issue90-fix \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    USER=app \
    MPLBACKEND=Agg \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Import the stack once (fails the build if something is missing). Run with a throw-away HOME/TMPDIR
# so nothing is left in the image's /tmp.
RUN export HOME=$(mktemp -d) TMPDIR=$(mktemp -d) MPLCONFIGDIR=$(mktemp -d) && python -c "import pynwb, oasis.functions; from cascade2p import cascade; \
from processor.cascade_infer import available_models; print(len(available_models()), 'CASCADE models')" \
    && rm -rf "$HOME" "$TMPDIR" "$MPLCONFIGDIR"

# Runs as uid 1000 on ECS and as an arbitrary uid on Lambda; nothing needs root. torch calls getpass:
# uid 1000 gets a passwd entry, any other uid gets USER=app.
RUN useradd --uid 1000 --no-create-home --home-dir /tmp --shell /usr/sbin/nologin processor
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
