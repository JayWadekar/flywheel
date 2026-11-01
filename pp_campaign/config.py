"""Paths for the O5 P-P campaign. Set them with environment variables:

    FLYWHEEL_O5_DATA     directory with the O5 catalog products (required): NSBH_cat_1yr_LVKpop_100realizations.h5
                         and the SNRts_..._O5_..._noise42_MLtemplate/ time series made by SNR_timeseries/
    FLYWHEEL_PSD_DIR     ASD files AplusDesign.txt, avirgo_O5low_NEW.txt (default: data/psds/O5 of this repo)
    FLYWHEEL_PP_RESULTS  output directory (default: pp_campaign/results)

Importing this module also puts SNR_timeseries/ and src/ of this repository on sys.path.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (os.path.join(REPO, 'SNR_timeseries'), os.path.join(REPO, 'src'), os.path.join(REPO, 'src', 'flywheel'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PSD_DIR = os.environ.get('FLYWHEEL_PSD_DIR', os.path.join(REPO, 'data', 'psds', 'O5'))
PSDS = {'L1': 'AplusDesign.txt', 'H1': 'AplusDesign.txt', 'Virgo': 'avirgo_O5low_NEW.txt'}
ASDS = {'H1': os.path.join(PSD_DIR, PSDS['H1']), 'L1': os.path.join(PSD_DIR, PSDS['L1']),
        'V1': os.path.join(PSD_DIR, PSDS['Virgo'])}
RESULTS = os.environ.get('FLYWHEEL_PP_RESULTS', os.path.join(HERE, 'results'))


def data_root():
    root = os.environ.get('FLYWHEEL_O5_DATA')
    if root is None:
        raise RuntimeError('Set FLYWHEEL_O5_DATA to the directory with the O5 catalog and SNR time series')
    return os.path.join(root, '')
