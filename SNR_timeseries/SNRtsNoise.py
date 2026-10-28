#
#    Gaussian noise generation in the Fourier domain, shared between the SNR time series pipeline
#    (compute_snrts_from_catalog.py) and the gwfast maximum likelihood scripts (e.g. calculate_scatter_from_catalog.py).
#
#    The noise realisation of a given event in a given detector depends only on
#    (parent_seed, catalog index of the event, detector name, absolute frequency), and NOT on the batching,
#    on which events pass the detection threshold, on fmin, or on the maximum frequency of the grid.
#    Any code using this module with the same parent seed, catalog index, detector name, df and PSD
#    gets exactly the same noise at the same frequencies.
#
#    Convention: the noise is used as d(f) = h(f) + n(f), with h in the gwfast convention
#    (i.e. complex conjugate of the LAL one), on linearly spaced grids f_k = k*df.
#    Each frequency bin has independent real and imaginary parts with standard deviation 0.5*sqrt(S_n(f)/df),
#    so that <|n(f)|^2> = S_n(f)/(2 df), as for a one-sided PSD.

import zlib
import numpy as np

def noise_rng(parent_seed, event_idx, det_name):
    '''
    Random number generator for a given event and detector.

    :param int parent_seed: The parent seed of the noise of the whole catalog.
    :param int event_idx: The index of the event in the catalog (position in the catalog file).
    :param str det_name: The name of the detector (e.g. ``'L1'``, or ``'ETS_0'`` for the arms of a triangle).

    :return: The random number generator.
    :rtype: numpy.random.Generator
    '''
    return np.random.default_rng(np.random.SeedSequence([int(parent_seed), int(event_idx), zlib.crc32(str(det_name).encode())]))

def white_noise(parent_seed, event_idx, det_name, nbins):
    '''
    Unit complex white noise on the bins k = 0, ..., nbins-1 (bin k corresponds to f = k*df).
    Real and imaginary parts are drawn alternately bin by bin, so the first m bins do not depend on ``nbins``.

    :param int parent_seed: The parent seed of the noise of the whole catalog.
    :param int event_idx: The index of the event in the catalog.
    :param str det_name: The name of the detector.
    :param int nbins: Number of frequency bins to generate.

    :return: Complex array of length ``nbins`` with real and imaginary parts ~ N(0, 1).
    :rtype: array
    '''
    w = noise_rng(parent_seed, event_idx, det_name).standard_normal((int(nbins), 2))
    return w[:, 0] + 1j*w[:, 1]

def generate_noise(freqs, psd_freq, psd_val, parent_seed, event_idx, det_name, df=None):
    '''
    Gaussian noise in the Fourier domain for a given event and detector, at the frequencies ``freqs``.

    :param array freqs: 1D frequency grid, linearly spaced with step df and with all f/df integers (e.g. fmin + k*df with fmin a multiple of df).
    :param array psd_freq: The frequencies at which the PSD is provided.
    :param array psd_val: The PSD values at ``psd_freq``.
    :param int parent_seed: The parent seed of the noise of the whole catalog.
    :param int event_idx: The index of the event in the catalog.
    :param str det_name: The name of the detector.
    :param float df: The frequency step, if None it is inferred from ``freqs``.

    :return: The noise at ``freqs``.
    :rtype: array
    '''
    freqs = np.asarray(freqs, dtype=float)
    if df is None:
        df = freqs[1] - freqs[0]
    kbins = np.rint(freqs/df).astype(int)
    if not np.allclose(kbins*df, freqs, rtol=0., atol=1e-6*df):
        raise ValueError('The frequency grid has to be of the form k*df with integer k (fmin has to be a multiple of df).')

    asd = np.interp(freqs, psd_freq, np.sqrt(psd_val), left=1., right=1.)
    w = white_noise(parent_seed, event_idx, det_name, kbins.max()+1)[kbins]

    return 0.5*asd/np.sqrt(df)*w
