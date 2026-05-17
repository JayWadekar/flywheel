"""Run a compact end-to-end test of the SNR-timeseries generation scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "example"
WORK_DIR = REPO_ROOT / "outputs" / "snr_timeseries_smoke_test"


def run(command: list[str]) -> None:
    print(" ".join(command))
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    input_dir = WORK_DIR / "input"
    ratios_dir = WORK_DIR / "mode_ratios"
    timeseries_dir = WORK_DIR / "snr_timeseries"
    for directory in (input_dir, ratios_dir, timeseries_dir):
        directory.mkdir(parents=True, exist_ok=True)

    catalog = input_dir / "catalog.hdf5"
    shutil.copy2(DATA_DIR / "snr_timeseries" / "detected_events_0_to_1.hdf5", catalog)
    snrs = input_dir / "snrs.txt"
    with h5py.File(catalog, "r") as h5:
        np.savetxt(snrs, h5["snr"][...])

    aligo_psd = DATA_DIR / "psds" / "LIGO-P1200087-v18-aLIGO_DESIGN.txt"
    adv_psd = DATA_DIR / "psds" / "LIGO-P1200087-v18-AdV_DESIGN.txt"
    common_args = [
        "--fname_obs",
        str(catalog),
        "--batch_size",
        "1",
        "--npools",
        "1",
        "--idx_in",
        "0",
        "--idx_f",
        "1",
        "--fmin",
        "20",
        "--fmax",
        "512",
        "--net",
        "H1",
        "L1",
        "Virgo",
        "--psds",
        str(aligo_psd),
        str(aligo_psd),
        str(adv_psd),
        "--lalargs",
        "HM",
        "--modes_list",
        "22",
        "33",
        "44",
        "--reference_detector",
        "L1",
        "--is_ASD",
        "True",
    ]

    run(
        [
            sys.executable,
            "SNR_timeseries/compute_mode_snr_ratios_from_catalog.py",
            "--fname_obs",
            str(DATA_DIR / "mode_ratios" / "sampled_GW190814_median_params_L1-H1-Virgo_O5_LAL-IMRPhenomXHM.h5"),
            "--batch_size",
            "100",
            "--npools",
            "1",
            "--idx_in",
            "0",
            "--idx_f",
            "1000",
            "--fmin",
            "20",
            "--fmax",
            "512",
            "--net",
            "H1",
            "L1",
            "Virgo",
            "--psds",
            str(aligo_psd),
            str(aligo_psd),
            str(adv_psd),
            "--lalargs",
            "HM",
            "--modes_list",
            "22",
            "33",
            "44",
            "--reference_detector",
            "L1",
            "--is_ASD",
            "True",
            "--fout",
            str(ratios_dir),
        ]
    )
    run(
        [
            sys.executable,
            "SNR_timeseries/compute_snrts_from_catalog.py",
            *common_args,
            "--fname_snrs",
            str(snrs),
            "--fout",
            str(timeseries_dir),
            "--snr_th",
            "0",
            "--time_interval",
            "4",
            "--df_integrals",
            "0.0009765625",
        ]
    )

    required = [
        ratios_dir / "snrs_22_L1_0_to_1000.txt",
        ratios_dir / "snrs_33_L1_0_to_1000.txt",
        ratios_dir / "snrs_44_L1_0_to_1000.txt",
        timeseries_dir / "snrs_timeseries_0_to_1.hdf5",
        timeseries_dir / "snrs_opt_info_0_to_1.hdf5",
        timeseries_dir / "detected_events_0_to_1.hdf5",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected outputs: {missing}")

    with h5py.File(timeseries_dir / "snrs_timeseries_0_to_1.hdf5", "r") as h5:
        event = h5["0"]
        assert set(event) == {"H1", "L1", "Virgo", "tgrid"}
        for detector in ("H1", "L1", "Virgo"):
            assert set(event[detector]) == {"22", "33", "44"}

    ratio_33 = np.loadtxt(ratios_dir / "snrs_33_L1_0_to_1000.txt")
    ratio_44 = np.loadtxt(ratios_dir / "snrs_44_L1_0_to_1000.txt")
    assert ratio_33.shape == (1000,)
    assert ratio_44.shape == (1000,)

    print(f"Smoke test passed. Outputs are in {WORK_DIR}")


if __name__ == "__main__":
    main()
