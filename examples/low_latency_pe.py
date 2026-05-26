# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: PE
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Low-latency PE with higher modes
#
# This jupytext notebook runs one event through the flywheel workflow:
#
# 1. load precomputed mode-by-mode SNR time series,
# 2. load sampled higher-mode amplitude ratios,
# 3. marginalize over extrinsic parameters with cogwheel,
# 4. compare 22-only and 22+higher-mode posterior samples.
#
# The repository does not include the large HDF5/text data products. Set the
# paths below to the directory where those products were generated.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flywheel import (
    SamplerSettings,
    build_dh_mtd,
    build_hh_md,
    compute_incoherent_lnprob,
    find_single_file,
    generate_samples_from_marg_info,
    load_intrinsic_samples,
    load_mode_ratio_samples,
    load_population,
    load_snr_opt_info,
    load_snr_timeseries,
    make_search_instance,
    postprocess_samples,
)


def show_keys(obj, depth=0, max_depth=3, max_items=7):
    """Print a compact map of nested dictionaries loaded from HDF5 files."""
    indent = "  " * depth
    if isinstance(obj, dict):
        keys = list(obj.keys())
        shown = keys[:max_items]
        suffix = " ..." if len(keys) > max_items else ""
        print(f"{indent}{type(obj).__name__}: {len(keys)} keys: {shown}{suffix}")
        if depth < max_depth:
            for key in shown:
                print(f"{indent}- {key}:")
                show_keys(obj[key], depth + 1, max_depth, max_items)
    else:
        print(f"{indent}{type(obj).__name__}")

# %%
plt.rcParams["figure.figsize"] = (5, 3)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.color"] = "#999999"
plt.rcParams["grid.linestyle"] = "--"

# %% [markdown]
# ## Configure paths and event

# %%
# When executed as a script, infer the repository root from this file.
# In a notebook converted with jupytext, __file__ may be unavailable; in that
# case run the notebook from the repository root or set REPO_ROOT manually.
try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd().parent
DATA_ROOT = REPO_ROOT / "data" / "example"
SNR_TIMESERIES_DIR = DATA_ROOT / "snr_timeseries"
SNR_RATIOS_DIR = DATA_ROOT / "mode_ratios"

# If cogwheel is not installed in the active environment, set this before
# running the notebook, for example:
# export FLYWHEEL_COGWHEEL_PATH=/path/to/cogwheel

EVENT_INDEX = 0
DETECTORS = ("H1", "L1", "V1")
RATIO_DETECTOR = "L1"
RATIO_STOP_KEY = "5000"
SAMPLES_PER_EVENT = 1000

# %% [markdown]
# ## Load precomputed inputs

# %%
snr_timeseries_path = find_single_file(SNR_TIMESERIES_DIR, "snrs_timeseries_*.hdf5")
snr_opt_info_path = find_single_file(SNR_TIMESERIES_DIR, "snrs_opt_info_*.hdf5")
events_path = find_single_file(SNR_TIMESERIES_DIR, "detected_events_*.hdf5")
intrinsic_path = find_single_file(SNR_RATIOS_DIR, "sampled*.h5")

snr_timeseries = load_snr_timeseries(snr_timeseries_path)
snr_opt_info = load_snr_opt_info(snr_opt_info_path)
events = load_population(events_path)
intrinsic = load_intrinsic_samples(intrinsic_path, samples_per_event=SAMPLES_PER_EVENT)
mode_ratio_samples = load_mode_ratio_samples(
    SNR_RATIOS_DIR,
    detector=RATIO_DETECTOR,
    stop_key=RATIO_STOP_KEY,
    samples_per_event=SAMPLES_PER_EVENT,
)

# The SNR time series dictionary is nested as:
# event index -> detector name -> mode -> complex SNR time series.
# Each event also stores a common time grid under the key "tgrid".
print("SNR time-series structure:")
show_keys(snr_timeseries, max_depth=2, max_items=5)

# The optimal-SNR dictionary is nested as:
# event index -> detector name -> mode -> <h_lm | h_lm>.
# The special "modes_correlation" entry stores normalized cross terms
# such as <h_22 | h_33> / sqrt(<h_22|h_22><h_33|h_33>).
print("SNR optimal-info structure:")
show_keys(snr_opt_info, max_depth=2, max_items=5)

# The event file stores source-frame masses, extrinsic parameters, aligned-spin
# components, and the detection SNR.  Add q and chi_eff for convenience.
events["q"] = events["m2_src"] / events["m1_src"]
events["chieff"] = (events["chi1z"] + events["q"] * events["chi2z"]) / (1 + events["q"])

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].hist(events["snr"], bins=100)
axes[0].set_title("SNR")
axes[1].hist(events["dL"], bins=100)
axes[1].set_title("dL")
axes[2].hist(events["q"], bins=100)
axes[2].set_title("q")
fig.tight_layout()

# %% [markdown]
# ## Build cogwheel inputs

# %%
settings = SamplerSettings(
    log2n_qmc=12,
    nphi=512,
    max_log2n_qmc=16,
    min_n_effective=50,
    n_qmc_sequences=100,
    seed=None,
)
cs_instance = make_search_instance(DETECTORS, settings=settings)

# In a search, every template waveform used for the SNR time series is
# normalized.  Cogwheel expects all inner products at its reference distance
# (1 Mpc), while the lookup table uses d_luminosity_max as the SNR=1 distance.
# This factor converts the precomputed normalized SNR products to that
# cogwheel convention.
dist_factor_ref = (
    cs_instance.lookup_table.d_luminosity_max
    / cs_instance.lookup_table.REFERENCE_DISTANCE
)

times = snr_timeseries[EVENT_INDEX]["tgrid"]

# dh_mtd is the matched-filter inner product (d|h_lm) for each mode,
# time sample, and detector.  Its dimensions are [mode, time, detector].
# The helper also maps V1 -> Virgo, matching the detector key used in the
# generated HDF5 files.
dh_mtd = build_dh_mtd(
    snr_timeseries,
    snr_opt_info,
    EVENT_INDEX,
    DETECTORS,
    dist_factor_ref,
)

# hh_md stores the mode-template covariance terms used by cogwheel.  The mode
# pair ordering is:
# <22,22>, <22,33>, <22,44>, <33,33>, <33,44>, <44,44>.
hh_md = build_hh_md(
    snr_opt_info,
    EVENT_INDEX,
    DETECTORS,
    dist_factor_ref,
)

# The incoherent proposal approximates the single-detector arrival-time
# likelihood.  Cogwheel uses it to propose times before enforcing coherent
# sky-location delays.
incoherent_lnprob_td = compute_incoherent_lnprob(
    dh_mtd,
    snr_opt_info,
    EVENT_INDEX,
    len(DETECTORS),
    dist_factor_ref,
)

print("dh_mtd shape:", dh_mtd.shape, "for [mode, time, detector]")
print("hh_md shape:", hh_md.shape, "for [mode-pair, detector]")
print("hh_md / dist_factor_ref^2:")
print(np.round(hh_md / dist_factor_ref**2, 2))

# %%
for i_det, detector in enumerate(DETECTORS):
    plt.plot(times, incoherent_lnprob_td[:, i_det], label=detector)
plt.xlabel("Time (s)")
plt.ylabel(r"Incoherent SNR$^2/2$")
plt.legend()

# %% [markdown]
# ## Run one event

# %%
rng = np.random.default_rng(seed=0)
mode_ratios_qm = mode_ratio_samples[EVENT_INDEX]
m2_samples_qm = intrinsic["m2_src"][EVENT_INDEX]
chieff_samples_qm = intrinsic["chieff"][EVENT_INDEX]

# The two columns of mode_ratios_qm are
# |h_physical_33|/|h_physical_22| and |h_physical_44|/|h_physical_22|.
# Rows are samples from the local intrinsic-parameter uncertainty estimate.
resampled = rng.choice(len(mode_ratios_qm), size=2**settings.max_log2n_qmc)
mode_ratios_qm = mode_ratios_qm[resampled]
m2_samples_qm = m2_samples_qm[resampled]
chieff_samples_qm = chieff_samples_qm[resampled]
print("Mode-ratio samples shape:", mode_ratios_qm.shape)

# %%
def run_event(
    cs_instance,
    dh_mtd,
    hh_md,
    times,
    incoherent_lnprob_td,
    mode_ratios_qm,
    n_runs=10,
    n_samples_per_run=1000,
):
    """Run higher-mode and quadrupole-only low-latency PE for one event.

    This is the crux of the workflow.  The inputs above summarize the data:

    - ``dh_mtd`` contains the mode-by-mode matched-filter SNR time series,
      ordered as [mode, time, detector].
    - ``hh_md`` contains the mode-template covariance terms.
    - ``incoherent_lnprob_td`` is the single-detector time proposal.
    - ``mode_ratios_qm`` contains samples of R_33 and R_44 for the intrinsic
      parameter-space neighborhood associated with the search template.

    The higher-mode run uses the sampled mode ratios.  The 22-only comparison
    sets those ratios to zero while keeping the same detector data and sampling
    settings, giving a BAYESTAR-like quadrupole-only baseline.
    """
    hm_samples = []
    qas_samples = []
    for _ in range(n_runs):
        # Marginalize over distance, phase, sky location, inclination,
        # polarization, geocentric time, and the discrete mode-ratio library.
        marg_info_hm = cs_instance.get_marginalization_info(
            dh_mtd,
            hh_md,
            times,
            incoherent_lnprob_td,
            mode_ratios_qm,
        )
        hm_samples.append(
            pd.DataFrame(
                generate_samples_from_marg_info(
                    cs_instance,
                    marg_info_hm,
                    n_samples_per_run,
                )
            )
        )

        # Repeat with R_33 = R_44 = 0 to remove higher-order-mode information.
        marg_info_22 = cs_instance.get_marginalization_info(
            dh_mtd,
            hh_md,
            times,
            incoherent_lnprob_td,
            mode_ratios_qm * 0,
        )
        qas_samples.append(
            pd.DataFrame(
                generate_samples_from_marg_info(
                    cs_instance,
                    marg_info_22,
                    n_samples_per_run,
                )
            )
        )

    return (
        pd.concat(hm_samples, ignore_index=True),
        pd.concat(qas_samples, ignore_index=True),
    )


# The 22-only run is generated by setting the higher-mode amplitude-ratio
# prior to zero inside run_event.  This gives a BAYESTAR-like quadrupole-only
# comparison with the same time-series inputs.
samples_hm, samples_22 = run_event(
    cs_instance,
    dh_mtd,
    hh_md,
    times,
    incoherent_lnprob_td,
    mode_ratios_qm,
    n_runs=10,
    n_samples_per_run=1000,
)

samples_hm, samples_22 = postprocess_samples(
    samples_hm,
    samples_22,
    snr_opt_info=snr_opt_info,
    events=events,
    event_index=EVENT_INDEX,
    m2_samples_qm=m2_samples_qm,
    chieff_samples_qm=chieff_samples_qm,
    redshift_reference=intrinsic["z_reference"][EVENT_INDEX],
    dist_factor_ref=dist_factor_ref,
)

# d_luminosity is returned in cogwheel-normalized units.  postprocess_samples
# converts it to Mpc, converts lon to RA using GMST, converts cos(iota) to the
# SNR-timeseries inclination convention, and attaches m2_src and chi_eff
# samples from the intrinsic-parameter samples.

# %% [markdown]
# ## Inspect results

# %%
truth = {
    "ra": events["ra"][EVENT_INDEX],
    "dec": events["dec"][EVENT_INDEX],
    "d_luminosity": events["dL"][EVENT_INDEX] * 1000,
    "psi": events["psi"][EVENT_INDEX],
    "iota": events["iota"][EVENT_INDEX],
    "m2_src": events["m2_src"][EVENT_INDEX],
    "chieff": events["chieff"][EVENT_INDEX],
    "t_geocenter": events["tGPS"][EVENT_INDEX],
}
pd.Series(truth, name="injection")

# %%
summary_keys = ["ra", "dec", "d_luminosity", "psi", "iota", "m2_src", "chieff", "lnl"]

def posterior_summary(samples, keys):
    table = samples[keys].quantile([0.05, 0.5, 0.95]).T
    table.columns = ["q05", "q50", "q95"]
    return table


summary = pd.concat(
    {
        "22-only": posterior_summary(samples_22, summary_keys),
        "22+HM": posterior_summary(samples_hm, summary_keys),
    },
    axis=1,
)
summary[("truth", "value")] = pd.Series(
    {key: truth.get(key, np.nan) for key in summary_keys}
)
summary

# %%
try:
    from cogwheel import gw_plotting

    plot_keys = ["ra", "dec", "d_luminosity", "psi", "iota", "m2_src", "chieff", "lnl"]
    gw_plotting.CornerPlot.DEFAULT_LATEX_LABELS["ra"] = r"${\rm RA}$"
    gw_plotting.CornerPlot.DEFAULT_LATEX_LABELS["dec"] = r"${\rm Dec}$"
    gw_plotting.CornerPlot.DEFAULT_LATEX_LABELS["m2_src"] = r"$m^{\rm src}_2 \ (M_{\odot})$"

    corner = gw_plotting.MultiCornerPlot(
        (samples_22[plot_keys], samples_hm[plot_keys]),
        tail_probability=1e-2,
        bins=40,
        labels=["22 only (BAYESTAR-like)", "22+HM"],
        smooth=3
    )
    corner.plot()
    corner.scatter_points(truth, adjust_lims=True, colors=["k"])
except ImportError:
    print("cogwheel plotting is unavailable; skipping corner plot.")

# %%
# Optional: save posterior samples for downstream plotting.
# output_path = REPO_ROOT / "outputs" / f"event_{EVENT_INDEX:04d}.hdf5"
# output_path.parent.mkdir(parents=True, exist_ok=True)
# with pd.HDFStore(output_path, mode="w") as store:
#     store["samples_22"] = samples_22
#     store["samples_HM"] = samples_hm
