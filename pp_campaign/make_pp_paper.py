"""Paper version of the P-P plot: the 22 + higher-harmonics run of one campaign, single panel, LaTeX labels in the
style of the Main.tex figures (Computer Modern, serif).

Usage: python make_pp_paper.py results_dir out.pdf
"""
import sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import make_pp_plot as mp

plt.rcParams.update({'text.usetex': True, 'font.family': 'serif', 'font.size': 15, 'axes.labelsize': 18,
                     'legend.fontsize': 13, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
LABELS = {'d_luminosity': r'$d_L$', 'iota': r'$\iota$', 'ra': r'$\alpha$', 'dec': r'$\delta$',
          'm2_src': r'$m_2^{\rm src}$'}


def p_text(p):
    if p >= 0.01:
        return f'{p:.2f}'
    m, e = f'{p:.0e}'.split('e')
    return rf'{m}\times10^{{{int(e)}}}'


if __name__ == '__main__':
    results_dir, out = sys.argv[1], sys.argv[2]
    cl, _ = mp.load(results_dir)
    cl = cl['HM']
    n = len(cl['iota'])
    fig, ax = plt.subplots(figsize=(6, 6))
    x = np.linspace(0, 1, 1001)
    for ci, alpha in [(0.68, 0.25), (0.95, 0.15), (0.997, 0.08)]:
        lo, hi = stats.binom.interval(ci, n, x)
        ax.fill_between(x, lo/n, hi/n, color='0.3', alpha=alpha, lw=0)
    for k in mp.PP_KEYS:
        v = np.sort(cl[k])
        p = stats.kstest(v, 'uniform').pvalue
        ax.plot(np.r_[0, v, 1], np.r_[0, np.arange(1, n + 1)/n, 1], color=mp.COLORS[k], lw=1.6,
                drawstyle='steps-post', label=rf'{LABELS[k]} ($p={p_text(p)}$)')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xlabel('Credible interval')
    ax.set_ylabel('Fraction of injections in credible interval')
    ax.legend(loc='upper left', title=rf'$N={n}$, KS $p$-values', title_fontsize=13, frameon=False)
    ax.grid(ls=':', lw=0.6, color='0.7')
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight')
    print('wrote', out, 'N =', n)
