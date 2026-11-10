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

For the computational analyst: F_corr = F − r·F_neu (r = 0.7, or `auto`: one r per session after
Dipoppa et al. 2018); F0 = Suite2p's maximin baseline
(Gaussian σ 10 frames, running min then max over 60 s) or a running 8th percentile; F0 is floored
at 10 % of the ROI's median raw F; ΔF/F = (F_corr − F0)/F0. CASCADE (CascadeTorch at c6978d5 plus
the upstream timing fix, below) runs on the accepted ROIs with a model chosen by indicator, frame
rate and noise level; OASIS (AR(1), decay `TAU`) with a minimum event size of 3 noise s.d. SNR = (99th
percentile of ΔF/F − median) / noise s.d. (from frame-to-frame differences); noise level as CASCADE
defines it (median |ΔF/F[t+1] − ΔF/F[t]| / √fs × 100).

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

`ophys-traces/<session_id>_ophys.nwb` (a new UUID `identifier`; the source NWB's name and identifier in `notes`; session_start_time, session_id, subject, keywords, ... copied), `processing/ophys`:

| Object | Content |
|---|---|
| `ImageSegmentation/PlaneSegmentation` | Every Suite2p ROI: `pixel_mask` (x = row = Suite2p's `ypix`, y = column = `xpix`, weight = Suite2p's `lam`: x is the first axis of the stored images, as in Suite2p's own NWB export and Allen's files), `iscell` (classifier probability), `accepted` (analysed downstream), `roi_id`, `plane`, `snr`, `noise_level`, `f0_floored`. |
| `Fluorescence/RoiResponseSeries`, `Fluorescence/Neuropil` | F and F_neu (Suite2p units). |
| `DfOverF/RoiResponseSeries` | ΔF/F (fraction). |
| `Deconvolved/RoiResponseSeries` | CASCADE spike rate (spikes/s, NWB unit `hertz`); NaN for ROIs not accepted (unless `CASCADE_ALL_ROIS=true`); NaN in the first and last ~32 frames, where CASCADE's window has no data. |
| `Deconvolved/OASIS` | OASIS event amplitudes (ΔF/F units), 0 between events. |
| `SummaryImages` | Suite2p's mean image, max projection and correlation map. |

All series are on the session clock (s): regular frame times (every interval within 0.1 % of the median, no gaps) are written as `starting_time` + `rate`, irregular ones as `timestamps` (one array, linked by the others). Planes k > 0 get a
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
| `NEUROPIL_COEFFICIENT` | `0.7` | r in F − r·F_neu, or `auto`: for each ROI, F_neu is binned into 20 intervals (equal-count bins in this implementation; the paper does not say which), the 5th percentile of F is taken in each, and the slope of a line through them is the ROI's r; the session's r is the mean over accepted cells whose raw F has skewness > 4 (sparse firers, whose lower envelope is their baseline; the median over all accepted ROIs when none is that skewed, where the paper used other experiments' average), clipped to 0-1.5 (the method of Dipoppa et al. 2018, after Peron et al. 2015). Per-ROI slopes are in `rois.csv` (`neuropil_alpha`); the r used and the median F_neu/F are in `summary.json`. |
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

Measured with the `cpu` image in Docker, amd64 emulation on an Apple M-series Mac (4 vCPU,
`--memory 4g --memory-swap 4g`), on the suite2p app's output for the full Allen demo (1,363 ROIs,
383 accepted, 9,405 frames at 15.04 Hz): **468 s wall, peak memory 2.3 GB** (cgroup `memory.peak`;
log in `data/work/allen-2p/logs/ophys-traces/`). An earlier run with other builds sharing the CPUs
took 1,607 s, of which CASCADE 1,468 s and OASIS 22 s: CASCADE dominates and grows with ROIs ×
frames (with `CASCADE_ALL_ROIS=true` it would be 3.5× longer here). Synthetic fixture (42 ROIs):
under 30 s. Memory is small: the traces (ROIs × frames, a few copies) plus CASCADE's input windows,
chunked to 512 MB.

GPU numbers: to be measured on the dev compute node.

## Results on the demo

Full Allen demo (`workflow-systems-neuroscience/data/work/allen-2p/ophys-traces/`, from the suite2p
app's Docker run; log in `data/work/allen-2p/logs/ophys-traces/`): 383 accepted cells; model
`Global_EXC_15Hz_smoothing100ms` (indicator GCaMP6f from the session NWB, 15 Hz, noise p90 4.1
within the model's 2-8); r = 0.7; median F_neu/F 0.92 (summary.json warns: strong neuropil); median
SNR 4.1; median noise level 2.6; mean CASCADE rate 0.59 spikes/s; OASIS 0.08 events/s; 26 ROIs
with a floored baseline. `NEUROPIL_COEFFICIENT=auto` gives r = 0.70 on this session (mean over 176
skewed cells), i.e. the default. Identity (start time, subject 496001363) copied from Allen's NWB, a
new identifier; times on the session clock (5.74-631.01 s). The masks overlay Allen's own masks
(39,668 shared pixels; 22,756 if transposed) and the bright cells of the mean image. nwb-inspect: pass, DANDI-ready
(no schema errors, no violations; 2 suggestions: experimenter and keywords are not in Allen's file).

**Sanity check against Allen** (not a validation: see Limitations): for the 201 cells that match an
Allen ROI (suite2p app, IoU > 0.5), our ΔF/F correlates with Allen's own `DfOverF` with median r =
0.87 (IQR 0.81-0.94), and our OASIS events (smoothed) with Allen's L0 `DfOverFEvents` with median
r = 0.89; our ΔF/F amplitudes (99th percentile) are a median 3.1× Allen's, the effect of r (Limitations). Synthetic fixture (planted spikes, one spike ≈ 0.2 ΔF/F): CASCADE rate vs planted spikes
median r > 0.85, mean rate within 10 % of the planted rate; OASIS r ≈ 0.78, event count ≈ spike count.

## Limitations

- **Absolute ΔF/F and CASCADE rates depend on the neuropil coefficient r.** F0 of F − r·F_neu
  shrinks as r grows, so ΔF/F (and with it CASCADE's noise level and rates) grows. A sweep
  on the Allen demo (amplitude relative to Allen's own DfOverF, CASCADE noise level): r = 0 → 1.12×,
  noise 0.87; r = 0.7 → 2.77×, noise 2.59. CASCADE expects a noise level of about 1 for Allen data.
  The demo's median F_neu/F is 0.92: strong neuropil, for which `summary.json` warns (above 0.8).
  Relative rates across cells and time are more robust than absolute ones; compare like with like,
  and consider `NEUROPIL_COEFFICIENT=auto` or a lower r when absolute rates matter.
- CASCADE's ground truth includes Allen Institute recordings: datasets 10-13 of its database are
  Huang, Ledochowitsch et al. (eLife 2021, doi:10.7554/eLife.51675; Allen Institute cell-attached
  recordings in GCaMP6 transgenic mice, related to DANDI:000048), and the `Global_EXC` models are
  trained on them (files `CAttached_Allen_Emx1_*` in CASCADE's `Ground_truth/`; numbered DS09-DS12 in
  the bundled models' `config.yaml` training lists). The demo session itself is not
  in it, but the lab, lines and pipeline overlap, so results on the Allen demo are a **sanity check,
  not a validation** of the spike rates.
- OASIS events are thresholded deconvolution, not detected action potentials; a burst is often one
  large event.
- Only accepted ROIs get CASCADE rates by default; rejected ROIs are kept (masks, F, ΔF/F, OASIS).
- Without a session NWB (or the suite2p app's session.json), the file's `session_start_time` is a
  placeholder (1970-01-01), it has no `subject` (nwb-inspect then reports it as not DANDI-ready:
  DANDI requires a subject), and times start at 0 when there are no timestamps. Link the session's
  NWB into `session`.

## Licences

GPL-3.0-or-later (`LICENSE`), because the app imports CascadeTorch and OASIS (both GPL-3.0)
in-process; CascadeTorch is modified only by the upstream timing fix (stated in `THIRD_PARTY.md`
and `patches/`). The CASCADE models are distributed with CascadeTorch. See `THIRD_PARTY.md`.

## Citations

Please cite these works if you use this app. CASCADE asks for the 2021 paper as the primary
reference, and the 2026 paper when GCaMP8 models are used. Verified on Crossref/DataCite and PubMed;
full author lists in `CITATION.cff`.

- Rupprecht P, Carta S, Hoffmann A, Echizen M, Blot A, Kwan AC, Dan Y, Hofer SB, Kitamura K, Helmchen F, Friedrich RW (2021). A database and deep learning toolbox for noise-optimized, generalized spike inference from calcium imaging. *Nature Neuroscience* 24:1324-1337. doi:10.1038/s41593-021-00895-5
- Rupprecht P, Rózsa M, Fang X, Svoboda K, Helmchen F (2026). Spike inference from calcium imaging data acquired with GCaMP8 indicators. *Nature Methods*. doi:10.1038/s41592-026-03183-x
- Friedrich J, Zhou P, Paninski L (2017). Fast online deconvolution of calcium imaging data. *PLOS Computational Biology* 13:e1005423. doi:10.1371/journal.pcbi.1005423, PMID: 28291787
- Stringer C, Ki C, Del Grosso N, LaFosse P, Zhang Q, Pachitariu M (2026). Extracting large-scale neural activity with Suite2p. *bioRxiv*. doi:10.64898/2026.02.04.703741
- Huang L, Ledochowitsch P, Knoblich U, Lecoq J, Murphy GJ, Reid RC, de Vries SE, Koch C, Zeng H, Buice MA, Waters J, Li L (2021). Relationship between simultaneously recorded spiking activity and fluorescence signal in GCaMP6 transgenic mice. *eLife* 10:e51675. doi:10.7554/eLife.51675, PMID: 33683198 (CASCADE ground truth that includes Allen recordings; see Limitations)
- With `NEUROPIL_COEFFICIENT=auto`: Dipoppa M, Ranson A, Krumin M, Pachitariu M, Carandini M, Harris KD (2018). Vision and Locomotion Shape the Interactions between Neuron Types in Mouse Visual Cortex. *Neuron* 98:602-615.e8. doi:10.1016/j.neuron.2018.03.037, PMID: 29656873
- With `NEUROPIL_COEFFICIENT=auto`, the method Dipoppa et al. adopted: Peron SP, Freeman J, Iyer V, Guo C, Svoboda K (2015). A Cellular Resolution Map of Barrel Cortex Activity during Tactile Behavior. *Neuron* 86:783-799. doi:10.1016/j.neuron.2015.03.027, PMID: 25913859
- Rübel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, Baker P, Soltesz I, Ng L, Svoboda K, Frank L, Bouchard KE (2022). The Neurodata Without Borders ecosystem for neurophysiological data science. *eLife* 11:e78362. doi:10.7554/eLife.78362, PMID: 36193886
- Software: CascadeTorch (Rupprecht P), https://github.com/PTRRupprecht/CascadeTorch, commit c6978d5; OASIS (Friedrich J), https://github.com/j-friedrich/OASIS, 0.3.2.
- Demo data: Baker C (2024). Allen Institute - Visual Coding - Optical Physiology. DANDI Archive. doi:10.48324/dandi.000728/0.240827.1809 (CC-BY-4.0); and de Vries SEJ, Lecoq JA, Buice MA, et al. (2020; full author list in `CITATION.cff` and the suite2p app's README). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. *Nature Neuroscience* 23:138-151. doi:10.1038/s41593-019-0550-9, PMID: 31844315

## Acknowledgements

CASCADE and CascadeTorch are developed by Peter Rupprecht in the Helmchen lab (University of Zurich)
with the co-authors above; OASIS by Johannes Friedrich, Pengcheng Zhou and Liam Paninski. The demo
data are the Allen Institute's Allen Brain Observatory (Visual Coding - Optical Physiology), shared
on the DANDI Archive; the output format is Neurodata Without Borders (NWB).
