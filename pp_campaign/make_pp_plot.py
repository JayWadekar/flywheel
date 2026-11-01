"""P-P plot for the O5 campaign from the per-event files written by run_pp_event.py.

Usage: python make_pp_plot.py [results_dir] [out_prefix] [--unweighted] [--raw] [--m2det]
--m2det adds the detector-frame m2 = m2_src (1 + z(d_L)) of the posterior samples (truth: m2_src (1 + z(d_L,true))).
Writes <out_prefix>.pdf/.png and prints KS p-values (per parameter and combined with Fisher's method).
"""
import sys, os, glob, json
import numpy as np, h5py
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PP_KEYS = ['d_luminosity', 'iota', 'ra', 'dec', 'm2_src']
LABELS = {'m2_det': r'$m_2^{\rm det}$', 'd_luminosity': r'$d_L$', 'iota': r'$\iota$', 'ra': 'RA', 'dec': 'Dec', 'm2_src': r'$m_2^{\rm src}$'}
COLORS = {'m2_det': '#8c564b', 'd_luminosity': '#1f77b4', 'iota': '#ff7f0e', 'ra': '#2ca02c', 'dec': '#d62728', 'm2_src': '#9467bd'}
RUN_TITLES = {'HM': '22 + higher harmonics', '22': '22 only (BAYESTAR-like)'}


def load(results_dir, key='cl', min_ess=100, suffix='', m2det=False):
    keys = PP_KEYS + ['m2_det'] if m2det else PP_KEYS
    if m2det:
        import cogwheel.cosmology
    cl = {r: {k: [] for k in keys} for r in ('HM', '22')}
    dropped = {'HM': 0, '22': 0}
    for fn in sorted(glob.glob(os.path.join(results_dir, 'event_*.h5'))):
        with h5py.File(fn, 'r') as f:
            for r in ('HM', '22'):
                g = f[r + suffix]
                if 'failed' in g.attrs or g.attrs['ess_w'] < min_ess:
                    dropped[r] += 1
                    continue
                c = json.loads(g.attrs[key])
                for k in PP_KEYS:
                    cl[r][k].append(c[k])
                if m2det:
                    tr = json.loads(f.attrs['truth'])
                    m2t = tr['m2_src']*(1 + float(cogwheel.cosmology.z_of_d_luminosity(tr['dL']*1000)))
                    m2d = g['m2_src'][:]*(1 + cogwheel.cosmology.z_of_d_luminosity(g['d_luminosity'][:]))
                    w = g['weight'][:].astype(float) if key == 'cl' else np.ones(len(m2d))
                    cl[r]['m2_det'].append(np.sum(w*(m2d < m2t))/w.sum())
    return {r: {k: np.array(v) for k, v in d.items()} for r, d in cl.items()}, dropped


def pp_panel(ax, cl, title):
    n = len(next(iter(cl.values())))
    x = np.linspace(0, 1, 1001)
    for ci, alpha in [(0.68, 0.25), (0.95, 0.15), (0.997, 0.08)]:
        lo, hi = stats.binom.interval(ci, n, x)
        ax.fill_between(x, lo/n, hi/n, color='0.3', alpha=alpha, lw=0)
    pvals = {}
    for k, v in cl.items():
        pvals[k] = stats.kstest(v, 'uniform').pvalue
        # KS p-value (small = inconsistent with the diagonal) and offset <Phi^-1(CL)> in sigma (> 0: posterior below the truth)
        p = pvals[k]
        ptxt = f'{p:.2f}' if p >= 0.01 else (f'{p:.0e}' if p >= 1e-99 else '<1e-99')
        off = stats.norm.ppf(np.clip(v, 1e-4, 1 - 1e-4)).mean()
        ax.plot(np.r_[0, np.sort(v), 1], np.r_[0, np.arange(1, n + 1)/n, 1], color=COLORS[k], lw=1.5,
                label=f'{LABELS[k]} (p = {ptxt}, {off:+.2f}$\\sigma$)', drawstyle='steps-post')
    combined = stats.combine_pvalues(list(pvals.values()), method='fisher').pvalue
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('C.I.'); ax.set_ylabel('Fraction of events in C.I.')
    ax.set_title(f'{title}\nN = {n}, combined p = {combined:.1e}', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, title=r'KS p-value, offset $\langle\Phi^{-1}({\rm C.I.})\rangle$', title_fontsize=9)
    return pvals, combined


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    results_dir = args[0]
    unweighted = '--unweighted' in sys.argv
    raw = '--raw' in sys.argv   # uncorrected distances (groups HM_raw, 22_raw) in the per-event files
    out = args[1] if len(args) > 1 else os.path.join(results_dir, 'pp_O5_ML' + ('_raw' if raw else '') + ('_unweighted' if unweighted else ''))
    m2det = '--m2det' in sys.argv
    cl, dropped = load(results_dir, 'cl_unweighted' if unweighted else 'cl', suffix='_raw' if raw else '', m2det=m2det)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, r in zip(axes, ('HM', '22')):
        pv, comb = pp_panel(ax, cl[r], RUN_TITLES[r] + (' [ML-template overlap]' if raw else '') + (' [no population weights]' if unweighted else ''))
        print(f'{r}: N={len(cl[r]["iota"])} (dropped {dropped[r]} with ESS<100)  KS p:',
              {k: round(v, 3) for k, v in pv.items()}, f' combined {comb:.3f}')
    fig.suptitle('O5 NSBH injections (noisy data, ML template), low-latency PE', fontsize=12)
    fig.tight_layout()
    fig.savefig(out + '.pdf', bbox_inches='tight')
    fig.savefig(out + '.png', bbox_inches='tight', dpi=110)
    print('wrote', out + '.pdf')
