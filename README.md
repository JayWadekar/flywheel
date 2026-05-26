# flywheel

`flywheel` is a version of the
[`cogwheel`](https://github.com/jroulet/cogwheel) library adapted for
low-latency neutron-star--black-hole (NSBH) parameter estimation.  Earlier
versions of this code were adapted for the coherent candidate-ranking statistic
in the [`IAS-HM`](https://github.com/JayWadekar/gwIAS-HM) search pipeline.

This repository provides a workflow to generate $(2,2)$ and $(2,2)$+higher-mode
inference for NSBH signals.  The code takes precomputed mode-by-mode SNR time
series and higher-mode amplitude-ratio samples, then uses `cogwheel`-style
marginalization over extrinsic parameters.

This repository is intended as a compact, public companion to the manuscript.
It includes a small GW190814-like example dataset.  Larger generated
HDF5/text products used for the population study are not checked into git.

## Repository layout

- `src/flywheel/`: reusable Python package.
- `SNR_timeseries/`: scripts used to generate mode-by-mode SNR time series and
  mode-amplitude ratio inputs from an event catalog.
- `examples/low_latency_pe.ipynb`: notebook showing one-event inference.
- `examples/low_latency_pe.py`: jupytext text version of the same notebook,
  useful for clean diffs and scripted execution.
- `data/example/`: small example inputs used by the notebook.
- `scripts/run_snr_timeseries_smoke_test.py`: compact generation test using the
  bundled example event.
- `pyproject.toml`: package metadata and Python dependencies.

## Installation

Create a fresh environment, then install the package in editable mode:

```bash
python -m pip install -e .
```

The core package depends on standard scientific Python packages.  The actual
marginalization also requires `cogwheel`, `gwpy`, and `lalsuite`:

```bash
python -m pip install -e ".[gw]"
```

Depending on your platform, `lalsuite` may be easier to install through conda.
If `cogwheel` is available as a local checkout rather than an installed
package, point `flywheel` to that checkout before running the example:

```bash
export FLYWHEEL_COGWHEEL_PATH=/path/to/cogwheel
```

## Included example data

The default notebook uses the files in `data/example/`.  This is a compact
one-event example containing:

- `data/example/snr_timeseries/snrs_timeseries_0_to_1.hdf5`
- `data/example/snr_timeseries/snrs_opt_info_0_to_1.hdf5`
- `data/example/snr_timeseries/detected_events_0_to_1.hdf5`
- `data/example/mode_ratios/sampled_GW190814_median_params_L1-H1-Virgo_O5_LAL-IMRPhenomXHM.h5`
- `data/example/mode_ratios/snrs_22_L1_0_to_5000.txt`
- `data/example/mode_ratios/snrs_33_L1_0_to_5000.txt`
- `data/example/mode_ratios/snrs_44_L1_0_to_5000.txt`
- `data/example/psds/LIGO-P1200087-v18-aLIGO_DESIGN.txt`
- `data/example/psds/LIGO-P1200087-v18-AdV_DESIGN.txt`

## Required input files for custom runs

For custom inputs, the workflow expects two directories:

1. A SNR time-series directory containing one file matching each pattern:
   - `snrs_timeseries_*.hdf5`
   - `snrs_opt_info_*.hdf5`
   - `detected_events_*.hdf5`

2. A mode-ratio directory containing:
   - `sampled*.h5`
   - `snrs_22_<detector>_0_to_<stop_key>.txt`
   - `snrs_33_<detector>_0_to_<stop_key>.txt`
   - `snrs_44_<detector>_0_to_<stop_key>.txt`

These are the outputs of the upstream SNR-timeseries and mode-ratio sampling
steps.  They are intentionally not checked into git because they are large
analysis products.

## Running the Example

The included notebook can be opened directly:

```bash
jupyter lab examples/low_latency_pe.ipynb
```

The paired jupytext file can also be run as a script:

```bash
python examples/low_latency_pe.py
```

For custom data, edit the path configuration near the top of
`examples/low_latency_pe.ipynb` or `examples/low_latency_pe.py`:

```python
DATA_ROOT = Path("/path/to/example_or_analysis_data")
SNR_TIMESERIES_DIR = DATA_ROOT / "snr_timeseries"
SNR_RATIOS_DIR = DATA_ROOT / "mode_ratios"
```

If you edit the `.py` version, regenerate the notebook with:

```bash
jupytext --to ipynb examples/low_latency_pe.py
```

## Regenerating SNR Time Series

The SNR generation scripts are included in `SNR_timeseries/`.  A compact
end-to-end smoke test uses the bundled one-event catalog and PSD files:

```bash
python scripts/run_snr_timeseries_smoke_test.py
```

The smoke test writes outputs under `outputs/snr_timeseries_smoke_test/`, which
is ignored by git.  It generates the SNR time series from the one-event catalog
and generates the higher-mode ratio files from the bundled intrinsic-sample
file.  The intrinsic-sample file itself is an upstream input to this step, not
an output of the SNR-timeseries scripts.

## Citation

If you use this workflow, please cite the accompanying manuscript and the
underlying `cogwheel`, `lalsuite`, and mode-by-mode filtering references.
