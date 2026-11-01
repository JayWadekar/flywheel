"""Intrinsic cloud weighted by the actual likelihood of the data, instead of the Gaussian (Fisher) approximation.

The likelihood in (Mc, eta, chi1z, chi2z) is banana-shaped at the O5 SNRs, so a Fisher ellipse puts weight
where the data rule it out. Here each proposal member theta_i gets the weight

    w_i ∝ exp(rho_i^2/2) p_pop(theta_i) / q(theta_i),   rho_i = max_{t, phase} |(d|h_i)| / sqrt(h_i|h_i),

with h_i the 22-mode template of theta_i at the ML extrinsic parameters (sky, psi, iota) and (d|h_i) the
network overlap with the data. This is the profile likelihood over amplitude, phase and time. Only the data and
the ML template enter, never the truth. The 22 mode carries the phasing (shape) information. The amplitude
information of the 33 and 44 modes is added by cogwheel through R33, R44 in the HM run, so using the HM template
here would count it twice.

The proposal q is adaptive importance sampling in N_ROUNDS rounds: first an inflated Gaussian from the (22-only,
spin-regularized) Fisher matrix at the ML, then inflated Gaussians fitted to all weighted draws so far. The
weights use the mixture of all the round proposals.

rho_i / rho_ML is also the factor by which a member's data overlap differs from the ML template's: the match used
in the distance correction of run_pp_event.py.
"""
import copy, os
import numpy as np

import config
import run_pp_event as rp
import flywheel.fisher as fisher

PSDS = config.PSDS
NOISE_SEED = 42
MODES_22 = [[2, 2], [2, -2]]
INFLATE = 1.5            # first-round proposal sigma / Fisher sigma
INFLATE_LATER = 1.3      # later rounds: proposal sigma / weighted-sample sigma
N_ROUNDS, N_ROUND = 6, 1000   # proposal rounds and draws per round
T_WINDOW = 5e-3          # s, time maximization window around the ML time
PAD = 2**20              # FFT length for the time maximization (dt = 1/(PAD df) ~ 8 us)


class DataLikelihood:
    """Noisy data of one O5 event (same signal and noise as the stored time series) and the 22-mode profile
    likelihood of templates at the ML extrinsic parameters."""

    def __init__(self, truth_ev, catalog_idx, ml):
        import SNRtsGlobals as gl
        import SNRtsML, waveforms as wf
        net = {k: copy.deepcopy(gl.detectors)[k] for k in PSDS}
        for k in net:
            net[k]['psd_path'] = os.path.join(config.PSD_DIR, PSDS[k])
        self.W = wf.LAL_WF('IMRPhenomXHM', is_HigherModes=True)
        self.W.fRef = rp.FMIN
        self.est = SNRtsML.ML_estimator(net, self.W, is_ASD=True, fmin=rp.FMIN, fmax=rp.FMAX)
        # The truth is only used to simulate the data
        self.f, self.data, _ = self.est.make_data(truth_ev, self.W, noise_seed=NOISE_SEED, event_idx=catalog_idx)
        self.df = self.f[1] - self.f[0]
        self.dets = list(net)
        self.invpsd = {k: 4*self.df/np.interp(self.f, self.est.PSDs[k][:, 0], self.est.PSDs[k][:, 1], left=1., right=1.)
                       for k in self.dets}
        self.base = dict(truth_ev)
        self.base.update({k: float(ml[k]) for k in ('Mc', 'eta', 'chi1z', 'chi2z', 'dL', 'iota', 'psi', 'ra', 'dec',
                                                     'Phicoal', 'tGPS')})
        self.proj = self.est.projection(self.base['ra'], self.base['dec'], self.base['psi'],
                                        SNRtsML.utils.GPSt_to_LMST(self.base['tGPS'], 0., 0.), self.f)
        tau = np.fft.fftfreq(PAD, d=self.df)       # time shifts of the ifft output, s
        self.win = np.where(np.abs(tau) < T_WINDOW)[0]

    def rho(self, Mc, eta, chi1z, chi2z):
        """Profile (amplitude, phase, time) network SNR of the 22-mode template."""
        p = dict(self.base, Mc=Mc, eta=eta, chi1z=chi1z, chi2z=chi2z)
        hp, hc = self.est.source(p, self.W, self.f, modes=MODES_22)
        h = self.est.combine(hp, hc, self.proj)
        hh = sum(np.sum(np.abs(h[k])**2*self.invpsd[k]) for k in self.dets)
        a = sum(self.data[k]*np.conj(h[k])*self.invpsd[k] for k in self.dets)
        z = np.abs(np.fft.ifft(a, n=PAD)*PAD)[self.win]
        j = np.argmax(z)
        if 0 < j < len(z) - 1:         # parabolic refinement of the peak
            y0, y1, y2 = z[j - 1]**2, z[j]**2, z[j + 1]**2
            den = y0 - 2*y1 + y2
            peak = y1 - (y2 - y0)**2/(8*den) if den < 0 else y1
        else:
            peak = z[j]**2
        return float(np.sqrt(peak/hh)) if hh > 0 else 0.


def _gauss_logpdf(x, mean, cov):
    L = np.linalg.cholesky(cov)
    u = np.linalg.solve(L, (x - mean).T)
    return -0.5*np.sum(u**2, axis=0) - np.sum(np.log(np.diag(L))) - 0.5*len(mean)*np.log(2*np.pi)


def _log_prior(x, z, pop):
    Mc, eta, c1, c2 = np.exp(x[:, 0]), x[:, 1], x[:, 2], x[:, 3]
    ok = (eta > 0.01) & (eta <= 0.25) & (np.abs(c1) < 0.99) & (np.abs(c2) < 0.99)
    out = np.full(len(x), -np.inf)
    pr = pop.chi('chi1', c1[ok])*pop.chi('chi2', c2[ok])*rp.mass_prior_lnMc_eta(Mc[ok], eta[ok], z, pop)
    with np.errstate(divide='ignore'):
        out[ok] = np.log(pr)
    return out


def likelihood_cloud(lik, p, F22, z, pop, rng, n=rp.N_CLOUD, n_rounds=N_ROUNDS, n_round=N_ROUND):
    """Cloud of n members in (Mc, eta, chi1z, chi2z), resampled with the 22 profile likelihood x population prior.

    Returns (cloud, info), cloud also holding 'rho' and 'match' = rho_i/rho_ML for each member."""
    G = F22.copy()
    G[2, 2] += 1/rp.SPIN_REG**2
    G[3, 3] += 1/rp.SPIN_REG**2
    mean0 = np.array([p['lnMc'], p['eta'], p['chi1z'], p['chi2z']])
    cov0 = INFLATE**2*fisher.intrinsic_covariance(G, 'marginal')
    rho_ml = lik.rho(np.exp(mean0[0]), *mean0[1:])

    def evaluate(x):
        lp = _log_prior(x, z, pop)
        r = np.zeros(len(x))
        for i in np.where(np.isfinite(lp))[0]:
            try:
                r[i] = lik.rho(np.exp(x[i, 0]), *x[i, 1:])
            except Exception:          # LAL domain errors at the edges of the parameter space
                lp[i] = -np.inf
        return lp, r

    # Adaptive importance sampling: each round draws from an inflated Gaussian fitted to all weighted draws so
    # far; the final weights use the equal-size mixture of all the round proposals (defensive mixture)
    props = [(mean0, cov0)]
    xs, lps, rs = [], [], []
    ess_rounds = []
    for k in range(n_rounds):
        mean, cov = props[-1]
        x = rng.multivariate_normal(mean, cov, size=n_round)
        lp, r = evaluate(x)
        xs.append(x); lps.append(lp); rs.append(r)
        x, lp, r = np.concatenate(xs), np.concatenate(lps), np.concatenate(rs)
        lq = np.logaddexp.reduce([_gauss_logpdf(x, m, c) for m, c in props], axis=0) - np.log(len(props))
        lw = lp + 0.5*r**2 - lq
        w = np.exp(lw - lw[np.isfinite(lw)].max())
        w[~np.isfinite(lw)] = 0
        ess_rounds.append(float(w.sum()**2/np.sum(w**2)))
        props.append((np.average(x, axis=0, weights=w), INFLATE_LATER**2*np.cov(x.T, aweights=w) + 1e-12*np.eye(4)))
    ess = ess_rounds[-1]
    idx = rng.choice(len(x), size=n, p=w/w.sum())
    cloud = {'Mc': np.exp(x[idx, 0]), 'eta': x[idx, 1], 'chi1z': x[idx, 2], 'chi2z': x[idx, 3],
             'rho': r[idx], 'match': r[idx]/rho_ml}
    info = {'lnl_cloud_ess': ess, 'rho22_ml_profile': rho_ml, 'rho22_max_cloud': float(r[w > 0].max()),
            'lnl_cloud_ess_rounds': ess_rounds,
            'sd_lnl_cloud': list(np.sqrt(np.diag(np.cov(x.T, aweights=w))))}
    return cloud, info
