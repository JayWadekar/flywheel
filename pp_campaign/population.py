"""Population prior of the O5 NSBH catalog (NSBH_cat_1yr_LVKpop_100realizations.h5), estimated
from the full 25M-event catalog and cached to population_prior.npz.

The catalog factorizes to good accuracy (|corr| < 0.06 except |chi1|-m1 = -0.10, which we ignore):
    p(m1_src, m2_src) p(chi1z) p(chi2z) p(dL) x isotropic (iota, sky, psi) x uniform time.
iota, psi, sky and time already match the cogwheel analysis priors; dL, masses and spins do not.
"""
import os
import numpy as np, h5py
from scipy.interpolate import RegularGridInterpolator
import config

CACHE = os.path.join(config.RESULTS, 'population_prior.npz')
DL_MAX_MPC = 8000.   # detected events are at dL < ~3 Gpc


def _smooth(h, n=3):
    k = np.ones(n)/n
    return np.convolve(np.pad(h, n//2, mode='edge'), k, mode='valid')


def build(cache=CACHE):
    with h5py.File(config.data_root() + 'NSBH_cat_1yr_LVKpop_100realizations.h5', 'r') as f:
        m1, m2 = f['m1_src'][:], f['m2_src'][:]
        chi1, chi2 = f['chi1z'][:], f['chi2z'][:]
        dL = f['dL'][:]*1000
    out = {}
    # Spins: 1D histograms on their supports (density, normalized)
    for name, x, lim in [('chi1', chi1, 1.0), ('chi2', chi2, 0.7)]:
        h, e = np.histogram(x, bins=400, range=(-lim, lim), density=True)
        out[name + '_edges'], out[name + '_dens'] = e, h
    # Masses: 2D histogram in (m1_src, m2_src)
    h, e1, e2 = np.histogram2d(m1, m2, bins=(200, 85), range=((2., 22.), (1., 2.7)), density=True)
    out['m1_edges'], out['m2_edges'], out['m12_dens'] = e1, e2, h
    # Distance (Mpc): histogram below DL_MAX_MPC, smoothed; normalized over the full catalog
    h, e = np.histogram(dL, bins=800, range=(0., DL_MAX_MPC))
    out['dL_edges'], out['dL_dens'] = e, _smooth(h.astype(float))/(len(dL)*np.diff(e))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, **out)
    return out


class PopulationPrior:
    def __init__(self, cache=CACHE):
        d = dict(np.load(cache)) if os.path.exists(cache) else build(cache)
        self.d = d
        c = lambda e: 0.5*(e[1:] + e[:-1])
        self._chi = {n: (d[n + '_edges'], d[n + '_dens']) for n in ('chi1', 'chi2')}
        self._m12 = RegularGridInterpolator((c(d['m1_edges']), c(d['m2_edges'])), d['m12_dens'],
                                            bounds_error=False, fill_value=0., method='linear')
        self._m_lims = (d['m1_edges'][[0, -1]], d['m2_edges'][[0, -1]])
        self._dL = (c(d['dL_edges']), d['dL_dens'])

    def chi(self, name, x):
        """Density of chi1z ('chi1') or chi2z ('chi2'); 0 outside the support."""
        e, h = self._chi[name]
        x = np.asarray(x)
        i = np.clip(np.searchsorted(e, x) - 1, 0, len(h) - 1)
        return np.where((x > e[0]) & (x < e[-1]), h[i], 0.)

    def chi_max(self, name):
        return self._chi[name][1].max()

    def chi_support(self, name):
        e = self._chi[name][0]
        return e[0], e[-1]

    def m12(self, m1, m2):
        """Density in (m1_src, m2_src); 0 outside the catalog's mass support."""
        pts = np.c_[np.ravel(m1), np.ravel(m2)]
        (a1, b1), (a2, b2) = self._m_lims
        inside = (pts[:, 0] > a1) & (pts[:, 0] < b1) & (pts[:, 1] > a2) & (pts[:, 1] < b2)
        out = np.zeros(len(pts))
        out[inside] = self._m12(pts[inside])
        return out.reshape(np.shape(m1))

    def dL(self, dL_mpc):
        x, h = self._dL
        return np.interp(dL_mpc, x, h, left=h[0], right=0.)


if __name__ == '__main__':
    pop = PopulationPrior()
    d = pop.d
    print('chi1 density at 0, 0.5, 0.95:', pop.chi('chi1', [0, .5, .95]), ' chi2 at 0, 0.5, 0.69:', pop.chi('chi2', [0, .5, .69]))
    print('dL density / dL^2 at 0.3, 1, 2, 3 Gpc (relative):', np.round(pop.dL([300, 1000, 2000, 3000])/np.array([300, 1000, 2000, 3000.])**2/(pop.dL(300)/300**2), 3))
