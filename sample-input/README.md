# sample-input

The input contract is in `workflow-systems-neuroscience/docs/conventions.md` ("Calcium imaging").

## Synthetic fixture (no download)

```sh
make cascade PYTHON=.venv/bin/python   # CascadeTorch + the CASCADE models into data/cascade/
make fixture PYTHON=.venv/bin/python   # data/fixture/{linked,flat}/
make run PYTHON=.venv/bin/python       # Docker over data/fixture/linked/ -> data/output/ophys-traces/
```

Two sessions written directly as Suite2p output: `sub-synth01_ses-day1` (the suite2p app's layout,
30 ROIs of which 24 are cells with planted Poisson spikes, 15 Hz, times from 100 s, with a session
NWB carrying identity and indicator GCaMP6f) and `mouse7_day3` (a lab's own
`labdata/mouse7_day3/suite2p/plane0/`, 30 Hz, no timestamps, named by `sessions.csv`).
`data/fixture/truth/` holds the planted spike trains.

## Real data: the Allen demo

Run the `suite2p` app on the Allen demo (its `sample-input/README.md`) and link its output into
`suite2p`; link the companion NWB
`sub-496001363/sub-496001363_ses-503109347-StimA_behavior+image+ophys.nwb` (DANDI:000728, asset
`d8e398ec-9b42-4369-a232-8d1ac7a0f80d`) into `session`. Ready-made suite2p output:
`workflow-systems-neuroscience/data/work/allen-2p/suite2p/`.

A lab's existing Suite2p folders work as they are (`<recording>/suite2p/plane0/`); add a
`sessions.csv` to name them, and the session NWB for identity.
