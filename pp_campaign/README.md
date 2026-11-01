# O5 P-P campaign: low-latency PE on noisy data

Code for the P-P tests of the low-latency (2,2)+HM pipeline on the simulated O5 NSBH population: noisy data, a
search-like maximum-likelihood (ML) template fitted to the noise, and an intrinsic-parameter cloud built for each event.
It produces the P-P figures of the paper (`figs/pp_O5_HM_fisher_refinement.pdf`, `figs/pp_O5_HM_flow.pdf`).

Two routes give the intrinsic cloud, i.e. the 1000 samples of (Mc, eta, chi1z, chi2z) with their mode ratios R33,
R44 and their match m = rho_i/rho_ML that feed the cogwheel marginalization:

- **Fisher + refinement** (`PP_CLOUD=rb`, default; ~6 s per event): Fisher matrix at the ML template, then adaptive
  importance sampling of the (2,2) likelihood with relative binning (`likelihood_cloud.py`, `rb_likelihood.py`).
  Needs the whitened data d(f).
- **Normalizing flow** (`PP_CLOUD=flow`; 0.2 s per event): a conditional flow trained on simulated searches draws the
  cloud from the search output alone (`flow/`).

## Files

| File | Purpose |
|---|---|
| `config.py` | Paths (environment variables below). |
| `o5_data.py` | Per-event access to the O5 time series; cogwheel inputs and marginalization. |
| `population.py` | Population prior estimated from the full catalog (cached to `RESULTS/population_prior.npz`). |
| `likelihood_cloud.py` | Noisy data of one event, (2,2) profile likelihood, adaptive importance sampling of the cloud. |
| `rb_likelihood.py` | Relative-binning likelihood and fast R33/R44. |
| `run_pp_event.py` | One event: cloud, cogwheel HM and 22-only runs, distance overlap correction, population weights, credible levels. |
| `run_pp_campaign.py` | Many events in parallel (event selection: rho_mf > 11, converged ML search). |
| `compute_rho_mf.py` | Matched-filter SNR of every event (needed for the selection). |
| `flow/make_training_set.py`, `flow/train_flow.py` | Training set from the simulated searches, and the conditional flow. |
| `make_pp_plot.py` | Diagnostic P-P plot (HM and 22-only panels, KS p-values and offsets; `--m2det` adds m2_det). |
| `make_pp_paper.py` | Paper version (HM panel). |

## Requirements

conda environment with `cogwheel`, `lalsuite`, `bilby`, `h5py`, `pandas`, `scipy`, `matplotlib`; for the flow also
`torch` and `zuko` (`pip install torch zuko`). A LaTeX installation for `make_pp_paper.py` (matplotlib `usetex`).

## Data and paths

```bash
export FLYWHEEL_O5_DATA=/path/to/nsbh_timeseries   # NSBH_cat_1yr_LVKpop_100realizations.h5 and
                                                   # SNRts_..._O5_..._noise42_MLtemplate/ (SNR_timeseries/compute_*.py)
export FLYWHEEL_PP_RESULTS=/path/to/results        # outputs (default: pp_campaign/results)
# optional: FLYWHEEL_PSD_DIR (default: data/psds/O5 of this repo: AplusDesign.txt, avirgo_O5low_NEW.txt)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker
```

The noisy data set is the one made with noise parent seed 42 and the ML templates of `SNR_timeseries/SNRtsML.py`
(`filter_params_*.hdf5`). The analysis regenerates each event's noisy strain from the same seed, so the cloud
likelihood sees exactly the data behind the stored time series.

## Reproducing the paper figures

```bash
cd pp_campaign
python compute_rho_mf.py                                  # RESULTS/rho_mf_ML.npy (~30 s)

# Fisher + refinement, 3000 events (~1 h on 40 cores)
PP_CLOUD=rb PP_OUT=$FLYWHEEL_PP_RESULTS/pp_O5_ML_rb python run_pp_campaign.py 3000 40
python make_pp_paper.py $FLYWHEEL_PP_RESULTS/pp_O5_ML_rb pp_O5_HM_fisher_refinement.pdf

# Flow: training set (~20 min on 48 cores), training with the 3000 P-P events held out (~3 min), campaign
python flow/make_training_set.py 48
python flow/train_flow.py $FLYWHEEL_PP_RESULTS/pp_O5_ML_rb/events_N3000_seed2026.txt
PP_CLOUD=flow PP_OUT=$FLYWHEEL_PP_RESULTS/pp_O5_ML_flow python run_pp_campaign.py 3000 40
python make_pp_paper.py $FLYWHEEL_PP_RESULTS/pp_O5_ML_flow pp_O5_HM_flow.pdf

# Diagnostic plots with both runs and m2_det
python make_pp_plot.py $FLYWHEEL_PP_RESULTS/pp_O5_ML_rb $FLYWHEEL_PP_RESULTS/pp_O5_ML_rb/pp --m2det
```

A single event: `PP_CLOUD=rb python run_pp_event.py 6 out_dir` (prints timings and credible levels).

The cloud is seeded per event (`1000 + iev`) and reproducible; cogwheel's QMC sequences are not seeded, so credible
levels vary at the ~0.01 level between reruns and the P-P offsets by ~0.002.

## Conventions

- Credible level of a parameter = weighted fraction of posterior samples below the true value (the bilby P-P
  convention). Offsets quoted as the mean of Phi^-1(credible level): 0 for a calibrated posterior, > 0 when the
  posterior lies below the truth (for d_L: too close).
- Posteriors are reweighted from the analysis priors (d_L^2; mass prior at z_ML) to the population prior.
- Events where the ML search fell back to the injection parameters (`ML_fallback = 1`) are excluded.

## Known limitations

- d_L is biased by about +0.07 to +0.08 sigma (posterior slightly too close). Known contributions: the (2,1) and (3,2) modes in the
  injections, which the 22+33+44 model lacks (~+0.02), and the noise-fitted ML template after the overlap correction
  (+0.02 HM, +0.006 22-only).
- The simulated search starts its optimizer at the true parameters, so its ML template sits ~0.5 posterior sigma from
  the truth. The Fisher + refinement posterior is set by the data likelihood and is unaffected; the flow learns this
  shortcut, so its m2 posteriors are too narrow. A flow for a real search must be trained on that search's outputs.
