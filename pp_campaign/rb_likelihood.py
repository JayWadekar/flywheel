"""Relative-binning (Zackay, Dai & Venumadhav 2018) version of likelihood_cloud.DataLikelihood.rho, and fast
mode-amplitude ratios R33, R44 on a sparse frequency grid. For the low-latency (~10 s) intrinsic cloud.

The reference is the ML template h0 (22 mode, ML extrinsic parameters). A template h = r(f) h0 has r(f) smooth,
so in each frequency bin b, r(f) ~ r0_b + r1_b (f - fm_b), and the overlaps reduce to per-bin summary data,
computed once per event:

    Z(t) = sum_det sum_b A0_b(t) r0_b* + A1_b(t) r1_b*,   A_b(t) = sum_{f in b} d h0* w (f - fm_b)^{0,1} e^{2 pi i f t}
    (h|h) = sum_det sum_b B0_b |r0_b|^2 + 2 B1_b Re(r0_b r1_b*) + B2_b |r1_b|^2

The time maximization is kept exact by putting e^{2 pi i f t} into the summary data on a grid of time shifts
(T_STEP over the same +-T_WINDOW as the exact code), followed by a parabolic refinement of the peak. A template
then needs the waveform only at the n_bins + 1 bin edges.
"""
import numpy as np

import lal
import lalsimulation as lalsim
import likelihood_cloud as lc
import SNRtsML
import SNRtsGlobals as gl
import SNRtsUtils as ut

GAMMAS = np.array([-5/3, -2/3, 1., 5/3, 7/3])   # PN powers of f in the phase
T_STEP = 5e-5                                     # s, time-shift grid of the summary data


def bin_edges(f, eps):
    """Bin edges (a subset of the grid f) such that a phase perturbation with every PN power, each at most
    2 pi at the band edge where it is largest, changes by at most ~eps within a bin (Zackay et al. 2018)."""
    fstar = np.where(GAMMAS > 0, f[-1], f[0])
    dphi = 2*np.pi*np.sum(np.sign(GAMMAS)[:, None]*(f[None, :]/fstar[:, None])**GAMMAS[:, None], axis=0)
    dphi -= dphi[0]
    n = int(np.ceil(dphi[-1]/eps))
    idx = np.unique(np.searchsorted(dphi, np.linspace(0, dphi[-1], n + 1)))
    idx = idx[idx < len(f)]
    if idx[-1] != len(f) - 1:
        idx = np.r_[idx, len(f) - 1]
    return idx


class RBLikelihood:
    """Same interface as likelihood_cloud.DataLikelihood (rho(Mc, eta, chi1z, chi2z)), with relative binning."""

    def __init__(self, lik, eps=0.3):
        self.lik = lik
        est, W, base = lik.est, lik.W, lik.base
        f = lik.f
        self.idx = bin_edges(f, eps)
        self.fe = f[self.idx]
        self.n_bins = len(self.fe) - 1
        hp, hc = est.source(base, W, f, modes=lc.MODES_22)
        h0 = est.combine(hp, hc, lik.proj)
        self.proj_e = est.projection(base['ra'], base['dec'], base['psi'],
                                     SNRtsML.utils.GPSt_to_LMST(base['tGPS'], 0., 0.), self.fe)
        # Templates at the bin edges come from SimInspiralChooseFDWaveformSequence with IMRPhenomXAS (= the XHM 22
        # mode, ~0.1 ms for 300 frequencies), but the data, the ML template and the exact likelihood use the
        # wrapper's dense-grid SimInspiralChooseFDWaveform, which differs by a match of ~0.997. The ratio of the
        # two at the ML template corrects the fast templates (|d lnL| <= 0.023 against the exact likelihood).
        self.corr = {k: 1. for k in lik.dets}
        h0_fast = self._strain(base)
        self.corr = {k: np.where(h0_fast[k] != 0, h0[k][self.idx]/np.where(h0_fast[k] != 0, h0_fast[k], 1), 0)
                     for k in lik.dets}
        self.h0_e = {k: h0[k][self.idx] for k in lik.dets}
        # Fine-grid points of bin b: idx[b] <= i < idx[b+1]; the last edge belongs to the last bin
        starts = self.idx[:-1]
        stop = np.r_[self.idx[1:-1], len(f)]
        fm = 0.5*(self.fe[:-1] + self.fe[1:])
        dfm = f - np.repeat(fm, stop - starts)
        self.t = np.arange(-lc.T_WINDOW, lc.T_WINDOW + T_STEP/2, T_STEP)
        self.A0, self.A1, self.B = {}, {}, {}
        for k in lik.dets:
            a = lik.data[k]*np.conj(h0[k])*lik.invpsd[k]
            A0 = np.empty((len(self.t), self.n_bins), complex)
            A1 = np.empty_like(A0)
            for j, tj in enumerate(self.t):
                v = a*np.exp(2j*np.pi*f*tj)
                A0[j] = np.add.reduceat(v, starts)
                A1[j] = np.add.reduceat(v*dfm, starts)
            self.A0[k], self.A1[k] = A0, A1
            b = np.abs(h0[k])**2*lik.invpsd[k]
            self.B[k] = [np.add.reduceat(b*dfm**n, starts) for n in range(3)]

    def _strain(self, p):
        """22-mode detector strains at the bin edges, same conventions as est.source + est.combine."""
        m1, m2 = ut.m1m2_from_Mceta(p['Mc'], p['eta'])
        v = lal.CreateREAL8Vector(len(self.fe))
        v.data = self.fe
        hp, hc = lalsim.SimInspiralChooseFDWaveformSequence(
            p['Phicoal'], m1*gl.uMsun, m2*gl.uMsun, 0., 0., p['chi1z'], 0., 0., p['chi2z'], self.lik.W.fRef,
            p['dL']*gl.uGpc, p['iota'], lal.CreateDict(), lalsim.IMRPhenomXAS, v)
        h = self.lik.est.combine(hp.data.data, hc.data.data, self.proj_e)
        return {k: h[k]*self.corr[k] for k in h}

    def rho(self, Mc, eta, chi1z, chi2z):
        """Profile (amplitude, phase, time) network SNR of the 22-mode template."""
        h = self._strain(dict(self.lik.base, Mc=Mc, eta=eta, chi1z=chi1z, chi2z=chi2z))
        z = np.zeros(len(self.t), complex)
        hh = 0.
        for k in self.lik.dets:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.where(self.h0_e[k] != 0, h[k]/self.h0_e[k], 0)
            r0 = 0.5*(r[:-1] + r[1:])
            r1 = (r[1:] - r[:-1])/np.diff(self.fe)
            z += self.A0[k] @ np.conj(r0) + self.A1[k] @ np.conj(r1)
            B0, B1, B2 = self.B[k]
            hh += np.sum(B0*np.abs(r0)**2 + 2*B1*np.real(r0*np.conj(r1)) + B2*np.abs(r1)**2)
        z2 = np.abs(z)**2
        j = np.argmax(z2)
        peak = z2[j]
        if 0 < j < len(z2) - 1:        # parabolic refinement of the peak
            den = z2[j - 1] - 2*z2[j] + z2[j + 1]
            if den < 0:
                peak = z2[j] - (z2[j + 1] - z2[j - 1])**2/(8*den)
        return float(np.sqrt(peak/hh)) if hh > 0 else 0.


def mode_ratios_fast(cloud, ml, n_freq=400):
    """R33, R44 (L1, iota = pi/2, h+ only, as SNRtsCompute.modeSNR_ratio) from the mode amplitudes on a
    log-spaced grid of n_freq frequencies (trapezoid rule), with direct SimInspiralChooseFDWaveformSequence calls
    (~0.2 ms each). The integrand |h_lm|^2/S is smooth, so a sparse grid suffices; the exact code sums over the
    full df = 1/8 Hz grid."""
    import run_pp_event as rp
    psd = np.loadtxt(rp.ASDS['L1'], usecols=(0, 1))
    fcut_par = 0.3/gl.GMsun_over_c3            # waveforms.LAL_WF.fcut: 0.3/(Mc G/c^3 eta^-3/5)
    out = np.zeros((len(cloud['Mc']), 2))
    pars = {}
    for l in (2, 3, 4):
        pars[l] = lal.CreateDict()
        ma = lalsim.SimInspiralCreateModeArray()
        lalsim.SimInspiralModeArrayActivateMode(ma, l, l)
        lalsim.SimInspiralModeArrayActivateMode(ma, l, -l)
        lalsim.SimInspiralWaveformParamsInsertModeArray(pars[l], ma)
    for i in range(len(out)):
        Mc, eta = float(cloud['Mc'][i]), float(cloud['eta'][i])
        m1, m2 = ut.m1m2_from_Mceta(Mc, eta)
        f = np.geomspace(rp.FMIN, min(fcut_par*eta**0.6/Mc, rp.FMAX), n_freq)
        v = lal.CreateREAL8Vector(n_freq)
        v.data = f
        invS = 1/np.interp(f, psd[:, 0], psd[:, 1]**2, left=1., right=1.)
        snr2 = [np.trapz(np.abs(lalsim.SimInspiralChooseFDWaveformSequence(
                    float(ml['Phicoal']), m1*gl.uMsun, m2*gl.uMsun, 0., 0., float(cloud['chi1z'][i]), 0., 0.,
                    float(cloud['chi2z'][i]), rp.FMIN, float(ml['dL'])*gl.uGpc, np.pi/2, pars[l],
                    lalsim.IMRPhenomXHM, v)[0].data.data)**2*invS, f) for l in (2, 3, 4)]
        out[i] = np.sqrt(snr2[1]/snr2[0]), np.sqrt(snr2[2]/snr2[0])
    return out
