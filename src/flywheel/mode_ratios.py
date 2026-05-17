"""Utilities for intrinsic samples and higher-mode amplitude ratios."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def eta_to_q(eta: np.ndarray) -> np.ndarray:
    """Convert symmetric mass ratio ``eta`` to mass ratio ``q <= 1``."""
    discriminant = np.sqrt(1.0 - 4.0 * eta)
    return (1.0 - discriminant) / (1.0 + discriminant)


def load_intrinsic_samples(path: str | Path, samples_per_event: int = 1000) -> dict:
    """Load source-frame secondary mass and effective-spin samples.

    The input file is expected to contain the ``Mc``, ``eta``, ``z``,
    ``chi1z`` and ``chi2z`` arrays produced by the mode-ratio sampling step.
    """
    with h5py.File(path, "r") as h5:
        eta = h5["eta"][:].reshape(-1, samples_per_event)
        q = eta_to_q(eta)
        redshift = h5["z"][:].reshape(-1, samples_per_event)
        mtot_source = (
            h5["Mc"][:] / (h5["eta"][:] ** 0.6) / (1.0 + h5["z"][:])
        ).reshape(-1, samples_per_event)
        chi1z = h5["chi1z"][:].reshape(-1, samples_per_event)
        chi2z = h5["chi2z"][:].reshape(-1, samples_per_event)

    return {
        "m2_src": q * mtot_source / (1.0 + q),
        "chieff": (chi1z + q * chi2z) / (1.0 + q),
        "z_reference": redshift[:, 0],
    }


def load_mode_ratio_samples(
    directory: str | Path,
    detector: str,
    stop_key: str,
    samples_per_event: int = 1000,
) -> np.ndarray:
    """Load ``|h_33|/|h_22|`` and ``|h_44|/|h_22|`` samples.

    Returns an array with shape ``(n_events, samples_per_event, 2)``.
    """
    directory = Path(directory)
    ratios = []
    for mode in ("33", "44"):
        numerator = np.loadtxt(directory / f"snrs_{mode}_{detector}_0_to_{stop_key}.txt")
        denominator = np.loadtxt(directory / f"snrs_22_{detector}_0_to_{stop_key}.txt")
        ratios.append((numerator / denominator).reshape(-1, samples_per_event))
    return np.moveaxis(np.asarray(ratios), 0, 2)
