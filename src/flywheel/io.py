"""Input/output utilities for the SNR-timeseries workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def find_single_file(directory: str | Path, pattern: str) -> Path:
    """Return the unique file matching ``pattern`` inside ``directory``."""
    matches = sorted(Path(directory).glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {directory}, "
            f"found {len(matches)}."
        )
    return matches[0]


def load_snr_timeseries(path: str | Path) -> dict[int, dict]:
    """Load mode-by-mode complex SNR time series from an HDF5 file."""
    loaded: dict[int, dict] = {}
    with h5py.File(path, "r") as h5:
        for event_key in h5:
            event_index = int(event_key)
            loaded[event_index] = {}
            group = h5[event_key]
            for key in group:
                node = group[key]
                if isinstance(node, h5py.Group):
                    loaded[event_index][key] = {
                        subkey: node[subkey][()] for subkey in node
                    }
                else:
                    loaded[event_index][key] = node[()]
    return loaded


def load_snr_opt_info(path: str | Path) -> dict[int, dict]:
    """Load per-detector optimal SNR and mode-correlation metadata."""
    loaded: dict[int, dict] = {}
    with h5py.File(path, "r") as h5:
        for event_key in h5:
            event_index = int(event_key)
            loaded[event_index] = {}
            group = h5[event_key]
            for key in group:
                loaded[event_index][key] = {
                    subkey: group[key][subkey][()][0] for subkey in group[key]
                }
    return loaded


def load_population(
    path: str | Path,
    n_events: int | None = None,
    keys_skip: Iterable[str] = (),
) -> dict[str, np.ndarray]:
    """Load event parameters from an HDF5 population file."""
    skip = set(keys_skip)
    events: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:
        for key in h5:
            if key not in skip:
                values = np.asarray(h5[key])
                events[key] = values[:n_events] if n_events is not None else values
    return events
