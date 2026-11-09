# Third-party software

**This repository is GPL-3.0-or-later** (see `LICENSE`): the app imports CascadeTorch (GPL-3.0) and
OASIS (`oasis-deconv`, GPL-3.0) in the same process, so it is a work based on them and is licensed
compatibly. Our code calls them as libraries; the only change to third-party code is the upstream
CASCADE spike-timing fix applied to CascadeTorch at build time (below, and
`patches/cascadetorch-issue90-off-by-one.patch`), which GPL-3.0 permits and which we state here as
it requires. Two images are built (`TORCH_FLAVOUR`): **cpu** (CPU-only PyTorch, no NVIDIA software)
and **gpu** (CUDA 12.6 PyTorch with NVIDIA's redistributable CUDA libraries).

| Component | What | Licence | Source |
|---|---|---|---|
| CascadeTorch, commit c6978d5ff33edad8792c76e040412f0636913092 (Peter Rupprecht; PyTorch port of CASCADE by Rupprecht, Helmchen and colleagues, University of Zurich) | Spike-rate inference (`cascade2p.cascade.predict`). Source files fetched at build time and checked by sha256; **modified**: two lines changed exactly as in CASCADE commit bcc00141a6a8 (HelmchenLabSoftware/Cascade issue #90, 2026-10-27: predictions were one frame late), and an empty `cascade2p/__init__.py` added (CascadeTorch issue #3). | GPL-3.0 | https://github.com/PTRRupprecht/CascadeTorch |
| CASCADE pretrained models (22 models, listed with sha256 in `cascade-models.txt`: `Global_EXC_*` at 3-40 Hz and `GC8_EXC_*_high_noise` at 3-40 Hz), by Peter Rupprecht / Helmchen lab | The trained networks CASCADE applies; downloaded from the links in CascadeTorch's `available_models_CascadeTorch.yaml` (drive.switch.ch) at build time. | Distributed with CascadeTorch (GPL-3.0) | https://github.com/PTRRupprecht/CascadeTorch |
| oasis-deconv 0.3.2 (Johannes Friedrich, Pengcheng Zhou, Liam Paninski) | OASIS event deconvolution. | GPL-3.0 | https://github.com/j-friedrich/OASIS |
| pynwb 4.2.0, hdmf 6.2.0 (NWB team, LBNL) | Writing the NWB file. | BSD-3-Clause | https://github.com/NeurodataWithoutBorders/pynwb |
| ruamel.yaml 0.19.1 | Reading the models' config.yaml (CascadeTorch). | MIT | PyPI |
| PyTorch 2.14.1 (`+cpu` / `+cu126`); triton 3.8.0 (gpu) | CASCADE's networks. | BSD-3-Clause style; triton MIT | https://pytorch.org |
| nvidia-*-cu12, cuda-toolkit 12.6.3, cuda-bindings, cuda-pathfinder (gpu variant only; `requirements-cuda.txt`) | CUDA runtime libraries. | NVIDIA proprietary, redistributable; Apache-2.0 | https://developer.nvidia.com/cuda-toolkit |
| NumPy 2.5.4, SciPy 1.18.1, h5py 3.16.0, pandas 3.1.1 | Arrays, filters, HDF5, tables. | BSD-3-Clause | PyPI |
| Matplotlib 3.11.2, Pillow 13.0.0 | Figures. | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org |
| jsonschema, jsonschema-specifications, referencing, rpds-py, attrs, platformdirs, python-dateutil, tzdata, contourpy, cycler, fonttools, kiwisolver, packaging, pyparsing, filelock, fsspec, Jinja2, MarkupSafe, mpmath, networkx, sympy, setuptools, typing_extensions | Dependencies of the above (versions in `requirements.txt`). | MIT / BSD / Apache-2.0 / PSF | PyPI |
| python:3.12.16-slim-bookworm | Base image. | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

## Notes on the models

- `GC8_EXC_15Hz_smoothing50ms_high_noise`'s download link serves a different model (its config.yaml
  names `GC8_EXC_10Hz_smoothing150ms_high_noise`); it is not bundled, `GC8_EXC_15Hz_smoothing100ms_high_noise`
  is. The build checks every model's config names the model it is filed under.

## Not bundled

- **Demo data**: DANDI:000728, Allen Institute, Visual Coding - Optical Physiology (CC-BY-4.0),
  doi:10.48324/dandi.000728/0.240827.1809.
