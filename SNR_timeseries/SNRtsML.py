#
#    Maximum likelihood estimation on noisy data, with the same signal model and the same noise realisation
#    as the SNR time series of SNRtsCompute.simulate_SNR_timeseries.injectSignal (noise from SNRtsNoise.py).
#    Port of the Levenberg-Marquardt ML search of gwfast's run/calculate_scatter_from_catalog.py, depending only on
#    the SNR_timeseries modules (no gwfast).
#
#    The ML parameters are found with a Levenberg-Marquardt (damped Gauss-Newton / Fisher-scoring) iteration
#    started at the true parameters:
#      * the template is split into the source-frame polarisations (LAL call, depend on the intrinsic parameters,
#        iota, Phicoal, dL) and the detector projection (cheap, depends on ra, dec, psi, tGPS), so that each Jacobian
#        column recomputes only what actually changes; the dL derivative is analytic (h ~ 1/dL);
#      * the other derivatives are central finite differences of the *same* template used in the likelihood, with
#        steps ml_fd_step/sqrt(F_pp) (a template change of noise-weighted norm ~ml_fd_step), set from a pilot pass;
#        parameters on which the template does not depend (zero Fisher diagonal in the pilot pass) are kept fixed;
#      * the step is a Levenberg-Marquardt step in Fisher-normalised coordinates with Nielsen damping update,
#        accepted only if the likelihood increases; numerically singular directions are dropped;
#      * convergence: predicted decrease of -2lnL < ml_tol, or actual decrease < ml_tol in two consecutive
#        iterations. ML_status: 1 converged, 2 stalled, 0 max iterations, 3 error during the search;
#      * if the search does not converge, the injection parameters are returned and ML_fallback is set to 1.
#    The time of arrival is fitted as an offset dtGPS (in seconds) from the true tGPS, to avoid round-off on GPS times.
#    The frequency grid and the inner product are the same as in injectSignal: f = fmin + k*df up to min(fcut, fmax),
#    (a|b) = 4 df sum Re(a b*)/S_n.

import copy
import numpy as np

import SNRtsGlobals as glob
import SNRtsUtils as utils
import SNRtsCompute as tsCompute
import SNRtsNoise as noise

# GMST (in days) advanced per SI second
GMST_RATE = tsCompute.GMST_RATE

ML_STATUS = {'maxiter': 0, 'converged': 1, 'stalled': 2, 'error': 3}

_PROJ_PARS = ('ra', 'dec', 'psi', 'dtGPS')

# Pilot finite-difference steps, deliberately tiny: only used to estimate the diagonal of the Fisher matrix
_PILOT_REL = {'Mc': 1e-10}
_PILOT_ABS = {'eta': 1e-9, 'ra': 1e-7, 'dec': 1e-7, 'iota': 1e-7, 'psi': 1e-7, 'dtGPS': 1e-8,
              'Phicoal': 1e-7, 'chi1z': 1e-8, 'chi2z': 1e-8, 'chi1x': 1e-8, 'chi2x': 1e-8,
              'chi1y': 1e-8, 'chi2y': 1e-8, 'Lambda1': 1e-3, 'Lambda2': 1e-3, 'ecc': 1e-8}

def modes_from_strings(modes):
    ''' ['22', '33'] -> [[2, 2], [2, -2], [3, 3], [3, -3]], as used for the individual modes in SNRtsCompute. '''
    if modes is None:
        return None
    out = []
    for mode in modes:
        l, m = int(mode[0]), int(mode[1])
        out += [[l, m], [l, -m]]
    return out

def fit_param_names(wf_model):
    names = ['Mc', 'eta', 'dL', 'ra', 'dec', 'iota', 'psi', 'dtGPS', 'Phicoal', 'chi1z', 'chi2z']
    if wf_model.is_Precessing:
        names += ['chi1x', 'chi2x', 'chi1y', 'chi2y']
    if wf_model.is_tidal:
        names += ['Lambda1', 'Lambda2']
    if wf_model.is_eccentric:
        names += ['ecc']
    return names

def _eig_normalised(F, g, rcond):
    '''
    Eigen-decomposition of the Fisher matrix in Fisher-normalised coordinates (Fn = F_pq/sqrt(F_pp F_qq)), dropping
    numerically singular directions (w < rcond*max w). Returns (ok, scale, V, w, gt) with gt = V^T (g/scale), or None.
    '''
    s = np.sqrt(np.clip(np.diag(F), 0., None))
    ok = s > 0.
    if not ok.any():
        return None
    Fn = F[np.ix_(ok, ok)]/np.outer(s[ok], s[ok])
    w, V = np.linalg.eigh(Fn)
    keep = w > rcond*w.max()
    return (ok, s[ok], V[:, keep], w[keep], V[:, keep].T @ (g[ok]/s[ok]))

def _lm_step(dec, P, lam):
    ''' Levenberg-Marquardt solution of (F + lam diag(F)) x = g from the _eig_normalised output. '''
    x = np.zeros(P)
    if dec is None:
        return x
    ok, s, V, w, gt = dec
    x[ok] = (V @ (gt/(w + lam)))/s
    return x

class ML_estimator(tsCompute.simulate_SNR_timeseries):
    '''
    ML search on noisy data for the detector network ``detNet`` (same format and PSD handling as
    :py:class:`SNRtsCompute.simulate_SNR_timeseries`).

    :param dict detNet: The detector network, as in simulate_SNR_timeseries.
    :param WaveFormModel waveform_rec: The recovery waveform model.
    :param list(str) modes_rec: Modes used in the recovery template (e.g. ``['22']``), None for all the modes of the model.
    :param float spin_prior: Maximum spin magnitude of each component.
    :param float chirpmass_prior: Maximum chirp mass.
    :param int ml_maxiter, float ml_tol, float ml_rcond, float ml_fd_step, int ml_max_linesearch, float ml_lm_lambda0: Settings of the LM search, as in calculate_scatter_from_catalog.py.
    '''
    def __init__(self, detNet, waveform_rec, is_ASD=False, fmin=2., fmax=4096., modes_rec=None,
                 spin_prior=0.999, chirpmass_prior=200.,
                 ml_maxiter=50, ml_tol=1e-3, ml_rcond=1e-12, ml_fd_step=1e-2, ml_max_linesearch=12, ml_lm_lambda0=1e-3,
                 verbose=False):
        super().__init__(detNet, is_ASD=is_ASD, fmin=fmin, fmax=fmax)
        self.waveform_rec = waveform_rec
        self.modes_rec = modes_from_strings(modes_rec)
        self.names = fit_param_names(waveform_rec)
        self.ip = {n: i for i, n in enumerate(self.names)}
        self.kind = {n: ('ana' if n == 'dL' else ('proj' if n in _PROJ_PARS else 'src')) for n in self.names}
        self.spin_prior = spin_prior
        self.chirpmass_prior = chirpmass_prior
        self.ml_maxiter = ml_maxiter
        self.ml_tol = ml_tol
        self.ml_rcond = ml_rcond
        self.ml_fdstep = ml_fd_step
        self.ml_nls = ml_max_linesearch
        self.ml_lam0 = ml_lm_lambda0
        self.ml_lam_min = 1e-12
        self.verbose = verbose

    #########################################################################
    # Signal model, same as SNRtsCompute.simulate_SNR_timeseries.injectSignal
    #########################################################################

    def frequency_grid(self, evParams, waveform, df=1./8.):
        fcut = waveform.fcut(**evParams)
        fcut = min(fcut, self.fmax)
        return np.arange(self.fmin, fcut + df, df)

    def source(self, evParams, waveform, f, modes=None):
        ''' Polarisations hp, hc without the time shift, as in injectSignal. '''
        pars = evParams
        if modes is not None:
            pars = dict(evParams)
            pars['modes'] = modes
        hp, hc = waveform.hphc(f, **pars)
        return np.asarray(hp, dtype=complex), -np.asarray(hc, dtype=complex)

    def projection(self, ra, dec, psi, tGMST, f):
        ''' {det: (F+ e^{i phase}, Fx e^{i phase})}, including the time shift and the delay from the Earth center, as in injectSignal. '''
        tshift = np.exp(1j*(2*np.pi*f*tGMST*3600.*24))
        out = {}
        for det in self.detNet.keys():
            tmpDeltLoc = utils.DeltLoc(self.lat_rad[det], self.long_rad[det], self.elevation[det], ra, dec, tGMST)
            Fp, Fc = utils.PatternFunction(self.lat_rad[det], self.long_rad[det], self.xax_rad[det], self.angbtwArms[det], ra, dec, tGMST, psi)
            common = tshift*np.exp(1j*2.*np.pi*f*tmpDeltLoc)
            out[det] = (Fp*common, Fc*common)
        return out

    @staticmethod
    def combine(hp, hc, proj):
        return {det: hp*proj[det][0] + hc*proj[det][1] for det in proj.keys()}

    def strains(self, evParams, waveform, f, modes=None, tGMST=None):
        ''' Detector strains for the event parameters evParams (which include tGPS). '''
        if tGMST is None:
            tGMST = utils.GPSt_to_LMST(evParams['tGPS'], 0., 0.)
        hp, hc = self.source(evParams, waveform, f, modes=modes)
        return self.combine(hp, hc, self.projection(evParams['ra'], evParams['dec'], evParams['psi'], tGMST, f))

    def make_data(self, evParams, waveform, noise_seed=None, event_idx=None, df=1./8.):
        '''
        Data d = h + n in each detector, with the same signal and noise as injectSignal(evParams, waveform, df, noise_seed, event_idx).

        :return: frequency grid, {det: data}, {det: signal}
        '''
        f = self.frequency_grid(evParams, waveform, df=df)
        h = self.strains(evParams, waveform, f)
        if noise_seed is None:
            return f, copy.deepcopy(h), h
        assert event_idx is not None, 'The catalog index of the event is needed to generate the noise.'
        d = {det: h[det] + noise.generate_noise(f, self.PSDs[det][:,0], self.PSDs[det][:,1], noise_seed, event_idx, det, df=df) for det in h.keys()}
        return f, d, h

    #########################################################################
    # ML search
    #########################################################################

    def _bounds(self, evParams):
        bnd = {'Mc': (0.5, self.chirpmass_prior), 'eta': (0.05, 0.25), 'dL': (1e-4, np.inf),
               'dec': (-0.5*np.pi, 0.5*np.pi), 'Lambda1': (0., np.inf), 'Lambda2': (0., np.inf), 'ecc': (0., 1.)}
        for k in ('chi1z', 'chi2z', 'chi1x', 'chi2x', 'chi1y', 'chi2y'):
            bnd[k] = (-self.spin_prior, self.spin_prior)
        lb = np.array([bnd.get(n, (-np.inf, np.inf))[0] for n in self.names])
        ub = np.array([bnd.get(n, (-np.inf, np.inf))[1] for n in self.names])
        return lb, ub

    def find_ML(self, evParams, f, data, df=1./8., h_true=None):
        '''
        Find the ML parameters for the data ``data`` on the grid ``f``, starting from the true parameters ``evParams``.

        :return: Dictionary with the ML parameters and diagnostics.
        :rtype: dict
        '''
        names, ip, P = self.names, self.ip, len(self.names)
        W = self.waveform_rec
        dets = list(self.detNet.keys())
        invpsd = {det: 4.*df/np.interp(f, self.PSDs[det][:,0], self.PSDs[det][:,1], left=1., right=1.) for det in dets}
        tGMST0 = utils.GPSt_to_LMST(evParams['tGPS'], 0., 0.)

        base = dict(evParams)
        if W.is_tidal and not ('Lambda1' in base and 'Lambda2' in base):
            base['Lambda1'], base['Lambda2'] = utils.Lam12_from_Lamt_delLam(evParams['LambdaTilde'], evParams['deltaLambda'], evParams['eta'])
        theta0 = np.array([0. if n == 'dtGPS' else float(base.get(n, 0.)) for n in names])
        lb, ub = self._bounds(evParams)
        theta0 = np.clip(theta0, lb, ub)

        def pars_of(th):
            p = dict(base)
            for i, n in enumerate(names):
                if n != 'dtGPS':
                    p[n] = th[i]
            return p

        def src_of(th):
            return self.source(pars_of(th), W, f, modes=self.modes_rec)

        def proj_of(th):
            return self.projection(th[ip['ra']], th[ip['dec']], th[ip['psi']], tGMST0 + GMST_RATE*th[ip['dtGPS']], f)

        def inner(a, b):
            return sum(np.sum(np.real(a[det]*np.conjugate(b[det]))*invpsd[det]) for det in dets)

        def chi2_of(H):
            c = sum(np.sum((np.abs(data[det] - H[det])**2)*invpsd[det]) for det in dets)
            return c if np.isfinite(c) else np.inf

        def jacobian(th, ep, src, proj, H, fd_idx):
            J = {det: np.zeros((len(f), P), dtype=complex) for det in dets}
            for det in dets:
                J[det][:, ip['dL']] = -H[det]/th[ip['dL']]
            for p in fd_idx:
                plus, minus = th.copy(), th.copy()
                plus[p] = min(th[p] + ep[p], ub[p])
                minus[p] = max(th[p] - ep[p], lb[p])
                dstep = plus[p] - minus[p]
                if dstep == 0.:
                    continue
                if self.kind[names[p]] == 'src':
                    Hp, Hm = self.combine(*src_of(plus), proj), self.combine(*src_of(minus), proj)
                else:
                    Hp, Hm = self.combine(*src, proj_of(plus)), self.combine(*src, proj_of(minus))
                for det in dets:
                    J[det][:, p] = (Hp[det] - Hm[det])/dstep
            return J

        def fisher_grad(J, r):
            F, g = np.zeros((P, P)), np.zeros(P)
            for det in dets:
                Jw = np.conjugate(J[det])*invpsd[det][:, None]
                F += np.real(Jw.T @ J[det])
                g += np.real(Jw.T @ r[det])
            return F, g

        def set_steps(F, ep):
            d = np.diag(F)
            good = np.isfinite(d) & (d > 0.)
            ep[good] = self.ml_fdstep/np.sqrt(d[good])
            return good

        # Current state at the true parameters + template consistency check
        TH = theta0.copy()
        src_c, proj_c = src_of(TH), proj_of(TH)
        H_c = self.combine(*src_c, proj_c)
        if h_true is not None:
            hh = inner(h_true, h_true)
            mism = sum(np.sum(np.abs(H_c[det] - h_true[det])**2*invpsd[det]) for det in dets)/(hh if hh > 0. else 1.)
        else:
            mism = np.nan
        c2_true = chi2_of(H_c)
        c2 = c2_true
        r = {det: data[det] - H_c[det] for det in dets}

        H_true, status, niter, fixed = H_c, ML_STATUS['maxiter'], 0, []
        try:
            # Pilot pass (tiny steps): sets the finite-difference steps; parameters the template does not depend on are fixed
            eps = np.array([_PILOT_REL[n]*abs(theta0[i]) if n in _PILOT_REL else _PILOT_ABS.get(n, 1e-8) for i, n in enumerate(names)])
            fd_idx = [ip[n] for n in names if self.kind[n] != 'ana']
            J = jacobian(TH, eps, src_c, proj_c, H_c, fd_idx)
            F, _ = fisher_grad(J, r)
            good = set_steps(F, eps)
            fd_idx = [p for p in fd_idx if good[p]]
            fixed = [names[p] for p in range(P) if not good[p]]
            if self.verbose and len(fixed) > 0:
                print('Parameters not affecting the template, kept fixed: %s' %fixed)

            status = ML_STATUS['maxiter']
            niter, nsmall = 0, 0
            lam, nu = self.ml_lam0, 2.
            for it in range(self.ml_maxiter):
                J = jacobian(TH, eps, src_c, proj_c, H_c, fd_idx)
                F, g = fisher_grad(J, r)
                del J
                set_steps(F, eps)
                dec = _eig_normalised(F, g, self.ml_rcond)
                d0 = _lm_step(dec, P, 0.)                       # undamped Gauss-Newton step
                pred0 = np.sum(g*d0)                            # its predicted decrease of -2lnL
                niter += 1
                if pred0 < self.ml_tol:
                    status = ML_STATUS['converged']
                    break

                c2_before = c2
                accepted = False
                for _ in range(self.ml_nls):
                    trial = np.clip(TH + _lm_step(dec, P, lam), lb, ub)
                    dd = trial - TH
                    st = src_of(trial) if any(dd[p] != 0. for p in range(P) if self.kind[names[p]] != 'proj') else src_c
                    pt = proj_of(trial)
                    Ht = self.combine(*st, pt)
                    ct = chi2_of(Ht)
                    # gain ratio: actual / predicted decrease of -2lnL for this step
                    predd = 2.*np.sum(g*dd) - dd @ F @ dd
                    gain = (c2 - ct)/predd if predd > 0. else 0.
                    if ct <= c2:
                        TH, c2, src_c, proj_c, H_c = trial, ct, st, pt, Ht
                        r = {det: data[det] - H_c[det] for det in dets}
                        # Nielsen (1999) damping update
                        rho = np.clip(gain, 0., 1.)
                        lam = max(lam*max(1./3., 1. - (2.*rho - 1.)**3), self.ml_lam_min)
                        nu = 2.
                        accepted = True
                        break
                    lam, nu = lam*nu, 2.*nu

                # second criterion: actual decrease below ml_tol in two consecutive iterations
                nsmall = nsmall + 1 if (accepted and (c2_before - c2) < self.ml_tol) else 0
                if nsmall >= 2:
                    status = ML_STATUS['converged']
                    break
                if not accepted:
                    status = ML_STATUS['stalled']
                    break
        except Exception as e:
            # e.g. LAL errors or non-finite templates during the search
            status = ML_STATUS['error']
            if self.verbose:
                print('WARNING: ML search failed with %s: %s' %(type(e).__name__, e))

        # Fallback: if the search did not converge, the injection parameters are returned (ML_fallback = 1)
        fallback = status != ML_STATUS['converged']
        if fallback:
            if self.verbose:
                print('WARNING: ML search not converged (status %s, %s iterations), falling back to the injection parameters.' %(status, niter))
            TH, c2, H_c = theta0.copy(), c2_true, H_true

        res = self._wrap_angles({n: TH[i] for i, n in enumerate(names)})
        res['tGPS'] = evParams['tGPS'] + res['dtGPS']
        # 2lnL(theta_ML) - 2lnL(theta_true): >= 0, mean ~ number of measurable parameter combinations
        res['ML_twoDeltalnL'] = c2_true - c2
        res['ML_niter'] = float(niter)
        res['ML_status'] = float(status)
        res['ML_fallback'] = float(fallback)
        res['ML_template_mismatch'] = mism
        res['ML_nfixed'] = float(len(fixed))
        # network matched-filter SNR of the ML template
        hh = inner(H_c, H_c)
        res['snr_ML_net'] = inner(data, H_c)/np.sqrt(hh) if hh > 0. else 0.
        return res

    def _wrap_angles(self, p):
        ''' Map angles back to their canonical ranges using exact symmetries of the signal. '''
        th = 0.5*np.pi - p['dec']
        c, s = np.cos(th), np.sin(th)
        p['dec'] = 0.5*np.pi - np.arccos(np.clip(c, -1., 1.))
        p['ra'] = np.mod(np.where(s < 0., p['ra'] + np.pi, p['ra']), 2.*np.pi)    # (theta, ra) ~ (-theta, ra+pi)
        p['psi'] = np.mod(p['psi'], np.pi)                                            # polarisation tensor is pi-periodic
        is_quadrupole = (not (self.waveform_rec.is_HigherModes or self.waveform_rec.is_Precessing)) or \
                        (self.modes_rec is not None and all(abs(m) == 2 and l == 2 for l, m in self.modes_rec))
        if is_quadrupole and not self.waveform_rec.is_Precessing:
            # quadrupole only: iota ~ -iota and the phase is 2pi-periodic
            p['iota'] = np.arccos(np.clip(np.cos(p['iota']), -1., 1.))
            p['Phicoal'] = np.mod(p['Phicoal'], 2.*np.pi)
        return {k: float(v) for k, v in p.items()}

    def __call__(self, evParams, waveform_inj, noise_seed=None, event_idx=None, df=1./8.):
        '''
        Build the noisy data for the event (same as injectSignal with the same noise_seed and event_idx) and find the ML parameters.

        :return: Dictionary with the ML parameters, diagnostics and SNRs.
        :rtype: dict
        '''
        missing_params = [p for p in waveform_inj.parameters if p not in evParams]
        assert not missing_params, f"Missing parameters: {missing_params}"
        f, d, h = self.make_data(evParams, waveform_inj, noise_seed=noise_seed, event_idx=event_idx, df=df)
        res = self.find_ML(evParams, f, d, df=df, h_true=h)
        # optimal and matched-filter (true template) SNRs
        num, den = 0., 0.
        for det in self.detNet.keys():
            invpsd = 4.*df/np.interp(f, self.PSDs[det][:,0], self.PSDs[det][:,1], left=1., right=1.)
            hh = np.sum(np.abs(h[det])**2*invpsd)
            hd = np.sum(np.real(d[det]*np.conjugate(h[det]))*invpsd)
            res['snr_opt_'+det] = np.sqrt(hh)
            res['snr_mf_'+det] = hd/np.sqrt(hh) if hh > 0. else 0.
            num, den = num + hd, den + hh
        res['snr_opt_net'] = np.sqrt(den)
        res['snr_mf_net'] = num/np.sqrt(den) if den > 0. else 0.
        return res
