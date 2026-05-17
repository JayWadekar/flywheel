"""Event-level low-latency PE workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SamplerSettings:
    """Numerical settings for the cogwheel marginalization integral."""

    log2n_qmc: int = 12
    nphi: int = 512
    max_log2n_qmc: int = 16
    min_n_effective: int = 50
    n_qmc_sequences: int = 100
    seed: int | None = None


def _detector_key(detector: str) -> str:
    """Map public detector labels to the labels stored in the HDF5 files."""
    return "Virgo" if detector == "V1" else detector


def make_search_instance(
    detectors: Sequence[str],
    settings: SamplerSettings = SamplerSettings(),
):
    """Create and randomize a higher-mode coherent-score instance."""
    from .coherent_score import SearchCoherentScoreHMAS, cogwheel

    sky_dict = cogwheel.likelihood.marginalization.SkyDictionary(
        "".join(detector[0] for detector in detectors)
    )
    instance = SearchCoherentScoreHMAS(
        sky_dict=sky_dict,
        log2n_qmc=settings.log2n_qmc,
        nphi=settings.nphi,
        max_log2n_qmc=settings.max_log2n_qmc,
        n_qmc_sequences=settings.n_qmc_sequences,
        seed=settings.seed,
    )
    instance.min_n_effective = settings.min_n_effective

    # Independent QMC sequences reduce Monte Carlo artifacts when several
    # marginalization runs are combined into one posterior sample set.
    for i in range(len(instance._qmc_sequences)):
        instance._rng.random()
        instance._qmc_sequences[i] = instance._create_qmc_sequence()
    return instance


def build_dh_mtd(
    snr_timeseries: dict,
    snr_opt_info: dict,
    event_index: int,
    detectors: Sequence[str],
    dist_factor_ref: float,
    reference_detector: str = "L1",
    modes: Sequence[str] = ("22", "33", "44"),
) -> np.ndarray:
    """Build ``(d|h_lm)`` with shape ``(mode, time, detector)``."""
    rows = []
    for mode in modes:
        mode_rows = []
        for detector in detectors:
            det_key = _detector_key(detector)
            series = snr_timeseries[event_index][det_key][mode]
            if det_key != reference_detector:
                scale = np.sqrt(
                    snr_opt_info[event_index][det_key][mode]
                    / snr_opt_info[event_index][reference_detector][mode]
                )
                series = series * scale
            mode_rows.append(series)
        rows.append(np.c_[mode_rows].T)
    return np.asarray(rows) * dist_factor_ref


def build_hh_md(
    snr_opt_info: dict,
    event_index: int,
    detectors: Sequence[str],
    dist_factor_ref: float,
    reference_detector: str = "L1",
) -> np.ndarray:
    """Build ``(h_lm|h_l'm')`` terms with shape ``(6, detector)``."""
    modes = ("22", "33", "44", "33", "44", "44")
    correlation_keys = (None, "22-33", "22-44", None, "33-44", None)
    rows = []
    for mode in modes:
        row = []
        for detector in detectors:
            det_key = _detector_key(detector)
            if det_key == reference_detector:
                row.append(1.0)
            else:
                row.append(
                    snr_opt_info[event_index][det_key][mode]
                    / snr_opt_info[event_index][reference_detector][mode]
                )
        rows.append(row)

    hh_md = np.asarray(rows, dtype=complex) * dist_factor_ref**2
    factors = [
        1 if key is None else 2 * snr_opt_info[event_index]["modes_correlation"][key]
        for key in correlation_keys
    ]
    return (hh_md.T * np.asarray(factors)).T


def compute_incoherent_lnprob(
    dh_mtd: np.ndarray,
    snr_opt_info: dict,
    event_index: int,
    n_detectors: int,
    dist_factor_ref: float,
) -> np.ndarray:
    """Compute the incoherent arrival-time proposal used by cogwheel."""
    covariance = np.eye(3, dtype=np.complex128)
    correlations = snr_opt_info[event_index]["modes_correlation"]
    covariance[0, 1] = correlations["22-33"]
    covariance[0, 2] = correlations["22-44"]
    covariance[1, 2] = correlations["33-44"]
    covariance[1, 0] = np.conj(covariance[0, 1])
    covariance[2, 0] = np.conj(covariance[0, 2])
    covariance[2, 1] = np.conj(covariance[1, 2])

    whitening = np.linalg.cholesky(np.linalg.inv(covariance)[::-1, ::-1])[::-1, ::-1]
    incoherent = np.zeros((dh_mtd.shape[1], n_detectors))
    for i_det in range(n_detectors):
        scores = dh_mtd[:, :, i_det].T / dist_factor_ref
        for i_time in range(dh_mtd.shape[1]):
            incoherent[i_time, i_det] = np.sum(
                np.abs(np.dot(whitening.T, scores[i_time])) ** 2
            ) / 2.0
    return incoherent


def generate_samples_from_marg_info(cs_instance, marg_info, num: int) -> dict:
    """Draw posterior samples from a cogwheel marginalization result."""
    sample_shape = np.full(num, np.nan)[()]
    if marg_info.q_inds.size == 0:
        return dict.fromkeys(
            [
                "d_luminosity",
                "dec",
                "lon",
                "phi_ref",
                "psi",
                "t_geocenter",
                "lnl_marginalized",
                "lnl",
                "h_h",
                "n_effective",
                "n_qmc",
                "cosiota",
                "q_ids",
            ],
            sample_shape,
        )

    random_ids = cs_instance._rng.choice(
        len(marg_info.q_inds), size=num, p=marg_info.weights
    )
    q_ids = marg_info.q_inds[random_ids]
    o_ids = marg_info.o_inds[random_ids]
    sky_ids = marg_info.sky_inds[random_ids]
    t_geocenter = (
        marg_info.t_first_det[random_ids]
        - cs_instance.sky_dict.geocenter_delay_first_det[sky_ids]
    )
    d_h = marg_info.d_h[random_ids]
    h_h = marg_info.h_h[random_ids]
    d_luminosity = cs_instance._sample_distance(d_h, h_h)
    distance_ratio = d_luminosity / cs_instance.lookup_table.REFERENCE_DISTANCE

    return {
        "d_luminosity": d_luminosity,
        "dec": cs_instance.sky_dict.sky_samples["lat"][sky_ids],
        "lon": cs_instance.sky_dict.sky_samples["lon"][sky_ids],
        "phi_ref": cs_instance._phi_ref[o_ids],
        "psi": cs_instance._qmc_sequence["psi"][q_ids],
        "t_geocenter": t_geocenter,
        "lnl_marginalized": np.full(num, marg_info.lnl_marginalized)[()],
        "lnl": d_h / distance_ratio - h_h / distance_ratio**2 / 2,
        "h_h": h_h / distance_ratio**2,
        "n_effective": np.full(num, marg_info.n_effective)[()],
        "n_qmc": np.full(num, marg_info.n_qmc)[()],
        "cosiota": cs_instance._qmc_sequence["cosiota"][q_ids],
        "q_ids": q_ids,
    }


def run_event(
    cs_instance,
    dh_mtd: np.ndarray,
    hh_md: np.ndarray,
    times: np.ndarray,
    incoherent_lnprob_td: np.ndarray,
    mode_ratios_qm: np.ndarray,
    n_runs: int = 10,
    n_samples_per_run: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run 22+HM and 22-only marginalization for one event."""
    hm_samples = []
    qas_samples = []
    for _ in range(n_runs):
        marg_info = cs_instance.get_marginalization_info(
            dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm
        )
        hm_samples.append(
            pd.DataFrame(generate_samples_from_marg_info(cs_instance, marg_info, n_samples_per_run))
        )

        marg_info_qas = cs_instance.get_marginalization_info(
            dh_mtd, hh_md, times, incoherent_lnprob_td, mode_ratios_qm * 0
        )
        qas_samples.append(
            pd.DataFrame(
                generate_samples_from_marg_info(cs_instance, marg_info_qas, n_samples_per_run)
            )
        )
    return pd.concat(hm_samples, ignore_index=True), pd.concat(qas_samples, ignore_index=True)


def postprocess_samples(
    samples_hm: pd.DataFrame,
    samples_22: pd.DataFrame,
    *,
    snr_opt_info: dict,
    events: dict,
    event_index: int,
    m2_samples_qm: np.ndarray,
    chieff_samples_qm: np.ndarray,
    redshift_reference: float,
    dist_factor_ref: float,
    reference_detector: str = "L1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert normalized samples to physical units and attach intrinsic samples."""
    try:
        import lal  # noqa: PLC0415
        import cogwheel  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("postprocess_samples requires lalsuite and cogwheel.") from exc

    hm = samples_hm.copy()
    qas = samples_22.copy()
    scale = (
        np.sqrt(snr_opt_info[event_index][reference_detector]["22"])
        * 2
        * events["dL"][event_index]
        * 1000
        / dist_factor_ref
    )
    hm["d_luminosity"] *= scale
    qas["d_luminosity"] *= scale

    hm["z"] = cogwheel.cosmology.z_of_d_luminosity(hm["d_luminosity"])
    qas["z"] = cogwheel.cosmology.z_of_d_luminosity(qas["d_luminosity"])
    hm["iota"] = np.pi - np.arccos(hm["cosiota"])
    qas["iota"] = np.pi - np.arccos(qas["cosiota"])

    intrinsic_samples = np.c_[m2_samples_qm, chieff_samples_qm]
    hm["m2_src"], hm["chieff"] = intrinsic_samples[hm["q_ids"].to_numpy()].T
    qas_indices = np.random.choice(len(intrinsic_samples), size=len(qas))
    qas["m2_src"], qas["chieff"] = intrinsic_samples[qas_indices].T

    hm["m2_src"] = hm["m2_src"] * (1 + redshift_reference) / (1 + hm["z"])
    qas["m2_src"] = qas["m2_src"] * (1 + redshift_reference) / (1 + qas["z"])

    gmst = [lal.GreenwichMeanSiderealTime(t) for t in hm["t_geocenter"].to_numpy()]
    hm["ra"] = cogwheel.skyloc_angles.lon_to_ra(hm["lon"], gmst)
    gmst = [lal.GreenwichMeanSiderealTime(t) for t in qas["t_geocenter"].to_numpy()]
    qas["ra"] = cogwheel.skyloc_angles.lon_to_ra(qas["lon"], gmst)
    return hm, qas
