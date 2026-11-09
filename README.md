# ophys-traces

This app infers spiking from calcium imaging with **[CASCADE](https://github.com/HelmchenLabSoftware/Cascade)**,
developed by **Peter Rupprecht** with Stefano Carta, Adrian Hoffmann, Mayumi Echizen, Antonin Blot,
Alex C. Kwan, Yang Dan, Sonja B. Hofer, Kazuo Kitamura, **Fritjof Helmchen** and **Rainer W.
Friedrich** (Brain Research Institute, University of Zurich, and Friedrich Miescher Institute, Basel;
Rupprecht et al. 2021, *Nature Neuroscience*, doi:10.1038/s41593-021-00895-5; GCaMP8 models:
Rupprecht, Rózsa, Fang, Svoboda & Helmchen 2026, *Nature Methods*, doi:10.1038/s41592-026-03183-x),
in its PyTorch port **[CascadeTorch](https://github.com/PTRRupprecht/CascadeTorch)** by Peter
Rupprecht, with CASCADE's pretrained models; and finds discrete events with **[OASIS](https://github.com/j-friedrich/OASIS)**
by **Johannes Friedrich, Pengcheng Zhou and Liam Paninski** (Columbia University; Friedrich et al.
2017, doi:10.1371/journal.pcbi.1005423). Its input comes from **Suite2p** (Stringer, Pachitariu et
al., HHMI Janelia; doi:10.64898/2026.02.04.703741). Pennsieve packages these methods; it did not
develop them, and the credit is their authors'. The demo data are from the **Allen Brain
Observatory** (DANDI:000728, doi:10.48324/dandi.000728/0.240827.1809; de Vries et al. 2020,
doi:10.1038/s41593-019-0550-9).

The second step of the **Two-photon Calcium Imaging** workflow:

- `suite2p → ophys-traces → ophys-activity → neuro-report`

Suite2p gives, for each cell, its raw fluorescence and that of the neuropil (the haze of out-of-focus
neurites around it). This app turns them into the signals neuroscientists analyse: **ΔF/F** (how
much brighter the cell is than its resting level), a **spike rate in spikes per second** from
CASCADE (a deep network trained on recordings where calcium imaging and electrophysiology were done
simultaneously, so its output is calibrated), and **discrete events** from OASIS. Everything is
written into one NWB file per session that also carries the cells' outlines and the session's
identity, ready for the next apps and for DANDI.

For the computational analyst: F_corr = F − r·F_neu (r = 0.7); F0 = Suite2p's maximin baseline
(Gaussian σ 10 frames, running min then max over 60 s) or a running 8th percentile; F0 is floored
at 10 % of the ROI's median raw F; ΔF/F = (F_corr − F0)/F0. CASCADE (CascadeTorch at c6978d5 plus
the upstream timing fix, below) runs on the accepted ROIs with a model chosen by indicator, frame
rate and noise level; OASIS (AR(1), decay `TAU`) with a minimum event size of 3 noise s.d.

## Ports

| Port | Direction | Content |
|---|---|---|
| `suite2p` | input (subfolder, required) | The suite2p app's output (`suite2p/<session_id>/plane<k>/` with `session.json`, `timestamps.npy`), or any Suite2p folder with `stat.npy`, `F.npy`, `Fneu.npy`, `ops.npy` (and `iscell.npy`), e.g. a lab's `<recording>/suite2p/plane0/`. Optional `sessions.csv`. |
| `session` | input (subfolder, optional) | The session's NWB (e.g. Allen's processed file, a DANDI asset), matched by `sub-<a>_ses-<b>` or `sessions.csv` (or the only NWB when there is one session). |
| `traces` | output | `ophys-traces/<session_id>_ophys.nwb`, `rois.csv`, figures, `summary.json`, `provenance.json`. |

**Sessions and time.** The session id comes from the suite2p app's `session.json`, else
`sessions.csv`, else `sub-<a>_ses-<b>` in the folder names, else the folder name (the folder above
`suite2p/` for a lab's own layout). Frame times are the suite2p app's `timestamps.npy` (session
clock); a folder without them gets `frame / ops["fs"]` from 0 s, and `summary.json` says so
(`time_source`). The NWB's identity (`session_start_time`, `identifier`, `session_description`,
`subject`, `timestamps_reference_time`, lab, institution) is copied from the session NWB, else from
the suite2p app's `session.json`; without either, `session_start_time` is set to 1970-01-01 with a
warning.

## Outputs

`ophys-traces/<session_id>_ophys.nwb`, `processing/ophys`:

| Object | Content |
|---|---|
| `ImageSegmentation/PlaneSegmentation` | Every Suite2p ROI: `pixel_mask` (x = column, y = row, weight = Suite2p's `lam`), `iscell` (classifier probability), `accepted` (analysed downstream), `roi_id`, `plane`, `snr`, `noise_level`, `f0_floored`. |
| `Fluorescence/RoiResponseSeries`, `Fluorescence/Neuropil` | F and F_neu (Suite2p units). |
| `DfOverF/RoiResponseSeries` | ΔF/F (fraction). |
| `Deconvolved/RoiResponseSeries` | CASCADE spike rate (spikes/s); NaN for ROIs not accepted (unless `CASCADE_ALL_ROIS=true`); 0 in the first and last ~32 frames, where CASCADE's window has no data. |
| `Deconvolved/OASIS` | OASIS event amplitudes (ΔF/F units), 0 between events. |
| `SummaryImages` | Suite2p's mean image, max projection and correlation map. |

All series have `timestamps` (s, session clock; one array, linked by the others). Planes k > 0 get a
`_plane<k>` suffix. `rois.csv`: `session_id, plane, roi_id, accepted, iscell_prob, x, y, npix,
radius, compact, snr, noise_level, f0_median, f0_floored, dff_max, cascade_rate_hz, oasis_events,
oasis_event_rate_hz` (+ `subject_id, condition, group` from the sheet). Figures:
`<session_id>_example_traces.png`, `_noise_levels.png`, `_snr.png` (captions in `summary.json`).
`summary.json` per session: `n_rois, n_accepted, frame_rate, n_frames, t_start, t_stop, time_source,
identity_source, indicator, cascade_model, cascade_model_reason, median_snr_accepted,
median_noise_level_accepted, mean_cascade_rate_hz_accepted, mean_oasis_event_rate_hz_accepted, ...`;
`provenance.json` (conventions.md keys plus `cascade`: the model, indicator and reason per session).

## CASCADE model choice

22 pretrained models are in the image (`cascade-models.txt`, each zip checked by sha256):
`Global_EXC_*` at 3, 5, 7.5, 10, 15, 20, 25, 30 and 40 Hz (with `_high_noise` versions at 3, 7.5,
15, 30 and 40 Hz), trained on CASCADE's whole excitatory-neuron ground truth (GCaMP6f/s, OGB-1,
Cal-520, R-CaMP, ...), and `GC8_EXC_*_high_noise` at 3-40 Hz, trained on GCaMP8 ground truth.
`CASCADE_MODEL=auto`:

1. **Family by indicator**: `INDICATOR=gcamp8`, or `auto` with a GCaMP8 indicator in the session
   NWB → `GC8_EXC`; `gcamp6f`, `gcamp6s`, `other`, or `auto` with any other or unknown indicator →
   `Global_EXC`. The Allen demo (GCaMP6f) uses `Global_EXC`.
2. **Frame rate**: the model rate closest to the recording's (log scale); ΔF/F is resampled to the
   model's rate when they differ by more than 5 % and the rate is interpolated back.
3. **Noise**: the standard model when its training noise levels cover the 90th percentile of the
   accepted ROIs' noise levels, else the `_high_noise` model.

The choice and the reason are logged and recorded in `summary.json` and `provenance.json`.

**The CascadeTorch timing fix.** CASCADE fixed a one-frame delay of its predictions on 2026-10-27
(HelmchenLabSoftware/Cascade issue #90, commit bcc00141a6a8). CascadeTorch's last commit (c6978d5,
2026-08-09) predates it and has the same two lines, so the image applies the same fix
(`patches/cascadetorch-issue90-off-by-one.patch`). Checked on synthetic traces: with the fix the
predicted rate peaks at the true spike frame (cross-correlation 0.392 at lag 0), without it one frame
late.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `NEUROPIL_COEFFICIENT` | `0.7` | r in F − r·F_neu. Allen estimates r per ROI (median 0.22 in the demo session); a lower r gives smaller ΔF/F. |
| `BASELINE` | `maximin` | `maximin` or `percentile`. |
| `BASELINE_WINDOW_S` | `60` | Baseline window (s). |
| `BASELINE_PERCENTILE` | `8` | For `percentile`. |
| `TAU` | `0` | Indicator decay (s) for OASIS; 0 = the Suite2p run's `tau`, else 0.7. |
| `INDICATOR` | `auto` | `auto`, `gcamp6f`, `gcamp6s`, `gcamp6`, `gcamp8`, `other` (model family, above). |
| `CASCADE_MODEL` | `auto` | Or one of the bundled model names. |
| `CASCADE_ALL_ROIS` | `false` | Also infer spikes for ROIs that are not accepted (slower). |
| `OASIS_THRESHOLD_SD` | `3` | Smallest OASIS event in noise s.d. |
| `CLASSIFIER_THRESHOLD` | empty | Empty = Suite2p's `iscell`; a number = accept ROIs with probability ≥ it. |
| `EXAMPLE_ROIS` | `8` | ROIs in the example-traces figure. |
| `SEED`, `THREADS`, `USE_GPU` | `42`, `0`, `false` (cpu) / `auto` (gpu) | |

## Variants

| Variant | Compute | When |
|---|---|---|
| `cpu` (default) | 4 vCPU, 4 GB | CASCADE on CPU takes about 0.35 s per ROI per 1,000 frames (measured natively on 4 threads; slower in emulation): minutes to tens of minutes for a typical session of a few hundred cells. Chosen as the default because the demo and most sessions fit, and the image is about 4× smaller than the CUDA one. |
| `gpu` | T4 or newer | Long recordings, thousands of ROIs or `CASCADE_ALL_ROIS=true`: CASCADE's small convolutional networks run many times faster on a GPU (research notes: < 1 min vs a few minutes for 300-500 neurons × 30k frames). Falls back to CPU when no GPU is visible. VRAM use is small (ROIs are processed in chunks of at most 512 MB of input windows). |

## Runtime

RUNTIME_TABLE

GPU numbers: to be measured on the dev compute node.

## Results on the demo

DEMO_RESULTS

## Limitations

- CASCADE's rates are calibrated on its ground truth. ΔF/F amplitudes depend on the neuropil
  coefficient and the baseline: with r = 0.7 on a recording with strong neuropil (the Allen demo),
  F0 is small and ΔF/F large, so absolute rates can be overestimated; relative rates across cells
  and time are more robust. Compare like with like.
- CASCADE's ground truth includes Allen Institute recordings: datasets 10-13 of its database are
  Huang, Ledochowitsch et al. (eLife 2021; Allen Institute cell-attached recordings in GCaMP6
  transgenic mice, the data behind DANDI:000048), and the `Global_EXC` models are trained on them
  (files `CAttached_Allen_Emx1_*` in CascadeTorch's `Ground_truth/`). The demo session itself is not
  in it, but the lab, lines and pipeline overlap, so results on the Allen demo are a **sanity check,
  not a validation** of the spike rates.
- CASCADE's README puts the Allen Visual Coding data at a noise level of about 1. With r = 0.7 the
  demo's ΔF/F (and with it the noise level) is about 3× larger than with Allen's own per-ROI r
  (median 0.22), because F0 of the neuropil-corrected trace is small: set NEUROPIL_COEFFICIENT
  lower for recordings with strong neuropil if you need absolute rates.
- OASIS events are thresholded deconvolution, not detected action potentials; a burst is often one
  large event.
- Only accepted ROIs get CASCADE rates by default; rejected ROIs are kept (masks, F, ΔF/F, OASIS).
- Without a session NWB (or the suite2p app's session.json), the file's `session_start_time` is a
  placeholder (1970-01-01) and times start at 0 when there are no timestamps.

## Licences

GPL-3.0-or-later (`LICENSE`), because the app imports CascadeTorch and OASIS (both GPL-3.0)
in-process; CascadeTorch is modified only by the upstream timing fix (stated in `THIRD_PARTY.md`
and `patches/`). The CASCADE models are distributed with CascadeTorch. See `THIRD_PARTY.md`.

## Citations

If you use this app, cite CASCADE and OASIS (CASCADE asks for the 2021 paper as the primary reference,
and the 2026 paper when GCaMP8 models are used) and the works below (also in `CITATION.cff`):

- Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, Kwan AC, Dan Y, Hofer SB, Kitamura K, Helmchen F, Friedrich RW (2021). A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. *Nature Neuroscience* 24:1324-1337. doi:10.1038/s41593-021-00895-5
- Rupprecht P, Rózsa M, Fang X, Svoboda K, Helmchen F (2026). Spike inference from calcium imaging data acquired with GCaMP8 indicators. *Nature Methods*. doi:10.1038/s41592-026-03183-x
- Friedrich J, Zhou P, Paninski L (2017). Fast online deconvolution of calcium imaging data. *PLOS Computational Biology* 13:e1005423. doi:10.1371/journal.pcbi.1005423, PMID: 28291787
- Stringer C, Ki C, Del Grosso N, LaFosse P, Zhang Q, Pachitariu M (2026). Extracting large-scale neural activity with Suite2p. *bioRxiv*. doi:10.64898/2026.02.04.703741
- Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L, Bouchard KE (2022). The Neurodata Without Borders ecosystem for neurophysiological data science. *eLife* 11:e78362. doi:10.7554/eLife.78362, PMID: 36193886
- Software: CascadeTorch (Rupprecht P), https://github.com/PTRRupprecht/CascadeTorch, commit c6978d5; OASIS (Friedrich J), https://github.com/j-friedrich/OASIS, 0.3.2.
- Demo data: Baker C (2024). Allen Institute - Visual Coding - Optical Physiology. DANDI Archive. doi:10.48324/dandi.000728/0.240827.1809 (CC-BY-4.0); and de Vries SEJ, Lecoq JA, Buice MA, et al. (2020; full author list in `CITATION.cff` and the suite2p app's README). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23:138-151. doi:10.1038/s41593-019-0550-9, PMID: 31844315
