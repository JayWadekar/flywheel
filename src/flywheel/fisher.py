"""Fisher-matrix estimate of the intrinsic-parameter uncertainty for aligned-spin
higher-mode waveforms.

The waveform and detector response come from bilby, so the conventions match
bilby full-PE runs.  The Fisher matrix is used to draw the local cloud of
(Mc, eta, chi1z, chi2z) samples whose 33/22 and 44/22 amplitude ratios feed the
coherent-score marginalization (see ``SNR_timeseries/compute_fisher_samples.py``).

Parameter vector (order of rows/columns in the Fisher matrix)::

    ln Mc, eta, chi1z, chi2z, ln dL, iota, psi, ra, dec, t_c, phi_c

``Mc`` is the detector-frame chirp mass, ``dL`` is in Mpc and ``t_c`` is the
geocentric GPS time.

Numerical notes
---------------
Derivatives use a 4th-order central stencil (analytic for ``ln dL`` and
``t_c``).  The default steps were chosen from a step scan for a GW190814-like
O5 event with IMRPhenomXHM and ``fmin = 10 Hz``: smaller steps hit the
waveform's numerical-noise floor, which spuriously over-constrains the
near-degenerate (eta, chi1z, chi2z) directions (this is what happens with
very small finite-difference steps, e.g. gwfast's numdifftools default for LAL
waveforms).  Check convergence with :func:`step_convergence` when the source
is very different.

Example::

    from flywheel.fisher import FisherHM, from_events, intrinsic_covariance, sample_cloud, aligned_spin_prior
    fm = FisherHM(asd_files={'H1': 'AplusDesign.txt', 'L1': 'AplusDesign.txt', 'V1': 'avirgo_O5low_NEW.txt'},
                  fmin=20., fmax=1024.)
    p = from_events(events, 0)
    cov = intrinsic_covariance(fm.fisher(p), 'marginal')
    cloud = sample_cloud(p, cov, 5000, spin_prior=aligned_spin_prior(0.6))
"""

from __future__ import annotations

import numpy as np

PARAMS = ['lnMc', 'eta', 'chi1z', 'chi2z', 'lndL', 'iota', 'psi', 'ra', 'dec', 'tc', 'phic']
INTRINSIC = PARAMS[:4]

DEFAULT_STEPS = {'lnMc': 1e-5, 'eta': 3e-5, 'chi1z': 1e-4, 'chi2z': 1e-3,
                 'iota': 1e-4, 'psi': 1e-4, 'ra': 1e-4, 'dec': 1e-4, 'phic': 3e-3}


def _bilby():
    try:
        import bilby  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError("flywheel.fisher requires bilby and lalsuite.") from exc
    bilby.core.utils.logger.setLevel('ERROR')
    return bilby


def eta_to_q(eta):
    """Mass ratio q <= 1 from the symmetric mass ratio."""
    discriminant = np.sqrt(1 - 4*eta)
    return (1 - discriminant) / (1 + discriminant)


def to_bilby(p):
    """Map a dict with keys PARAMS to bilby's BBH parameter names."""
    return {'chirp_mass': np.exp(p['lnMc']),
            'mass_ratio': eta_to_q(p['eta']),
            'chi_1': p['chi1z'], 'chi_2': p['chi2z'],
            'luminosity_distance': np.exp(p['lndL']),
            'theta_jn': p['iota'], 'psi': p['psi'],
            'ra': p['ra'], 'dec': p['dec'],
            'geocent_time': p['tc'], 'phase': p['phic']}


def from_events(events, index=0):
    """Build the PARAMS dict from a gwfast-style event dictionary.

    Expected keys: ``Mc`` (detector frame, Msun), ``eta``, ``chi1z``, ``chi2z``,
    ``dL`` (Gpc), ``iota``, ``psi``, ``ra``/``dec`` (or ``phi``/``theta``),
    ``tGPS`` and ``Phicoal``.
    """
    ev = {k: np.atleast_1d(v)[index] for k, v in events.items()}
    dec = ev['dec'] if 'dec' in ev else np.pi/2 - ev['theta']
    ra = ev['ra'] if 'ra' in ev else ev['phi']
    return {'lnMc': np.log(ev['Mc']), 'eta': ev['eta'],
            'chi1z': ev['chi1z'], 'chi2z': ev['chi2z'],
            'lndL': np.log(ev['dL']*1000), 'iota': ev['iota'], 'psi': ev['psi'],
            'ra': ra, 'dec': dec, 'tc': ev['tGPS'], 'phic': ev['Phicoal']}


class FisherHM:
    """Network Fisher matrix for aligned-spin waveforms (IMRPhenomXHM by default).

    Parameters
    ----------
    asd_files : dict
        Detector name (bilby convention: 'H1', 'L1', 'V1', ...) -> ASD file
        (two columns: frequency, ASD).
    fmin, fmax, df : float
        Frequency range and resolution of the inner products (Hz).
    fref : float, optional
        Reference frequency, defaults to ``fmin``.
    mode_array : list, optional
        LAL mode array, e.g. ``[[2, 2], [2, -2]]`` for a 22-only Fisher matrix.
    multibanding : bool
        Keep IMRPhenomXHM's multibanding (default off for cleaner derivatives).
    """

    def __init__(self, asd_files, fmin=10., fmax=4096., df=1/8., fref=None,
                 approximant='IMRPhenomXHM', mode_array=None, multibanding=False):
        bilby = _bilby()
        self.fmin, self.fmax, self.df = fmin, fmax, df
        self.duration = 1/df
        self.sampling_frequency = 2*fmax
        waveform_arguments = {'waveform_approximant': approximant,
                              'reference_frequency': fref or fmin,
                              'minimum_frequency': fmin,
                              'maximum_frequency': fmax}
        if mode_array is not None:
            waveform_arguments['mode_array'] = mode_array
        if not multibanding:
            waveform_arguments['PhenomXHMThresholdMband'] = 0.
        self.wfg = bilby.gw.waveform_generator.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.sampling_frequency,
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)
        self.ifos = bilby.gw.detector.InterferometerList(list(asd_files))
        for ifo in self.ifos:
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=asd_files[ifo.name])
            ifo.minimum_frequency, ifo.maximum_frequency = fmin, fmax
        self._segment_tc = None

    def _set_segment(self, tc):
        # Merger 2 s before the end of the segment; only frequency-domain
        # inner products are used, so the exact placement is irrelevant.
        if self._segment_tc == tc:
            return
        for ifo in self.ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=self.sampling_frequency,
                                                duration=self.duration, start_time=tc - self.duration + 2)
        self.freqs = self.ifos[0].frequency_array
        self.masks = [ifo.frequency_mask for ifo in self.ifos]
        self.psds = [ifo.power_spectral_density_array for ifo in self.ifos]
        self._segment_tc = tc

    def strains(self, p):
        """Detector strains h_d(f) (masked to [fmin, fmax]) for parameters ``p``."""
        self._set_segment(p['tc'])
        bp = to_bilby(p)
        pols = self.wfg.frequency_domain_strain(bp)
        return [ifo.get_detector_response(pols, bp)[m] for ifo, m in zip(self.ifos, self.masks)]

    def inner(self, a, b, i_det):
        return 4*np.real(np.sum(a*np.conj(b)/self.psds[i_det][self.masks[i_det]]))*self.df

    def snr(self, p):
        """Optimal SNR in each detector."""
        h = self.strains(p)
        return np.sqrt([self.inner(hd, hd, i) for i, hd in enumerate(h)])

    def derivatives(self, p, steps=None):
        steps = {**DEFAULT_STEPS, **(steps or {})}
        h0 = self.strains(p)
        dh = {}
        for name in PARAMS:
            if name == 'lndL':
                dh[name] = [-hd for hd in h0]
            elif name == 'tc':
                # A geocentric time shift is a pure phase in every detector
                dh[name] = [-2j*np.pi*self.freqs[m]*hd for hd, m in zip(h0, self.masks)]
            else:
                s = steps[name]
                hp1, hm1, hp2, hm2 = [self.strains({**p, name: p[name] + k*s}) for k in (1, -1, 2, -2)]
                dh[name] = [(8*(a - b) - (c - d))/(12*s) for a, b, c, d in zip(hp1, hm1, hp2, hm2)]
        return dh

    def fisher(self, p, steps=None, detectors=None):
        """Fisher matrix summed over all detectors (or the named subset)."""
        dh = self.derivatives(p, steps)
        use = [i for i, ifo in enumerate(self.ifos) if detectors is None or ifo.name in detectors]
        n = len(PARAMS)
        F = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                F[i, j] = F[j, i] = sum(self.inner(dh[PARAMS[i]][d], dh[PARAMS[j]][d], d) for d in use)
        return F


def invert(F):
    """Invert a Fisher matrix after normalizing by its diagonal; returns (cov, condition number)."""
    s = np.sqrt(np.diag(F))
    Fn = F/np.outer(s, s)
    return np.linalg.inv(Fn)/np.outer(s, s), np.linalg.cond(Fn)


def intrinsic_covariance(F, kind='marginal'):
    """4x4 covariance of (ln Mc, eta, chi1z, chi2z).

    ``'marginal'``: intrinsic block of F^-1 (extrinsic parameters marginalized).
    ``'conditional'``: inverse of the intrinsic block of F (extrinsic parameters fixed).
    """
    k = len(INTRINSIC)
    if kind == 'marginal':
        return invert(F)[0][:k, :k]
    if kind == 'conditional':
        return invert(F[:k, :k])[0]
    raise ValueError(kind)


def step_convergence(fm, p, scales=(3, 2, 1, 0.5, 1/3)):
    """Marginal sigmas for all PARAMS when every finite-difference step is scaled.

    Returns an array (len(scales), len(PARAMS)) normalized to the ``scale = 1`` row.
    """
    sig = np.array([np.sqrt(np.diag(invert(fm.fisher(p, steps={k: v*s for k, v in DEFAULT_STEPS.items()}))[0]))
                    for s in scales])
    return sig/sig[list(scales).index(1)]


def aligned_spin_prior(a_max=0.6):
    """bilby's AlignedSpin prior (a ~ U(0, a_max), cos tilt ~ U(-1, 1))."""
    bilby = _bilby()
    return bilby.gw.prior.AlignedSpin(a_prior=bilby.core.prior.Uniform(0, a_max),
                                      z_prior=bilby.core.prior.Uniform(-1, 1))


def sample_intrinsic(p, cov, n=5000, rng=None):
    """Gaussian (Mc, eta, chi1z, chi2z) samples around ``p``, redrawn inside 0 < eta <= 0.25, |chi| < 1."""
    rng = np.random.default_rng(rng)
    mean = np.array([p['lnMc'], p['eta'], p['chi1z'], p['chi2z']])
    out = np.empty((0, 4))
    while len(out) < n:
        x = rng.multivariate_normal(mean, cov, size=2*n)
        x = x[(x[:, 1] > 0) & (x[:, 1] <= 0.25) & (np.abs(x[:, 2]) < 1) & (np.abs(x[:, 3]) < 1)]
        out = np.r_[out, x]
    out = out[:n]
    return {'Mc': np.exp(out[:, 0]), 'eta': out[:, 1], 'chi1z': out[:, 2], 'chi2z': out[:, 3]}


def sample_cloud(p, cov, n, spin_prior=None, rng=None):
    """Fisher (Gaussian likelihood) samples times a spin prior, by rejection sampling.

    ``spin_prior`` is a bilby 1D prior applied independently to chi1z and chi2z,
    or None (only physical bounds are imposed).
    """
    rng = np.random.default_rng(rng)
    out = {k: np.empty(0) for k in ['Mc', 'eta', 'chi1z', 'chi2z']}
    if spin_prior is not None:
        lo, hi = spin_prior.minimum, spin_prior.maximum
        wmax = np.max(spin_prior.prob(np.linspace(lo, hi, 20001)[1:-1]))**2
        # AlignedSpin's density diverges at chi=0; bilby's interpolated prob caps it, other priors may not
        assert np.isfinite(wmax) and wmax > 0, 'spin prior density must be bounded for rejection sampling'
    while len(out['Mc']) < n:
        s = sample_intrinsic(p, cov, n=20*n, rng=rng)
        if spin_prior is not None:
            inside = (s['chi1z'] > lo) & (s['chi1z'] < hi) & (s['chi2z'] > lo) & (s['chi2z'] < hi)
            w = np.zeros(len(inside))
            w[inside] = spin_prior.prob(s['chi1z'][inside])*spin_prior.prob(s['chi2z'][inside])
            keep = rng.random(len(w)) < w/wmax
            s = {k: v[keep] for k, v in s.items()}
        for k in out:
            out[k] = np.r_[out[k], s[k]]
    return {k: v[:n] for k, v in out.items()}
