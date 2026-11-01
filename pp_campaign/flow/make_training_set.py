"""Training set for an amortized intrinsic-parameter posterior: for every row of Francesco's O5 'ML' data set, the
search output (ML template parameters, matched-filter SNR) and, for the TRUE template of that event, the quantities
the low-latency run needs per sample:

    inputs : ln Mc, eta, chi1z, chi2z of the ML template, ln dL_ML, rho_mf (22 peak network SNR of the filtered series)
    targets: ln Mc, eta, chi1z, chi2z (true), R33 = sqrt(opt33/opt22), R44 = sqrt(opt44/opt22) in L1 (iota = pi/2),
             match = rho(true template)/rho(ML template), the profile 22 SNRs at the ML extrinsic parameters
             (likelihood_cloud.DataLikelihood.rho, on the noisy data with noise seed 42).

Over events, the true parameters are draws from p(theta | search output), so a conditional density model of the
targets given the inputs is a calibrated posterior by construction (neural posterior estimation).
The training set covers every row; train_flow.py excludes the P-P events (its HOLDOUT list) from the fit.

Usage: python make_training_set.py [n_procs] [n_rows]  -> RESULTS/flow/training_set.npz (needs rho_mf_ML.npy)
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from multiprocessing import Pool
import config

OUT = os.path.join(config.RESULTS, 'flow', 'training_set.npz')
INTR = ('Mc', 'eta', 'chi1z', 'chi2z')
_d = {}


def data():
    if not _d:
        import o5_data as od
        _d['d'] = od.O5Data('ML')
        _d['rho_mf'] = np.load(os.path.join(config.RESULTS, 'rho_mf_ML.npy'))
    return _d['d'], _d['rho_mf']


def mode_snr_ratios(lik, Mc, eta, c1, c2):
    """R33, R44 of a template: sqrt(opt_mode/opt_22) in L1, h+ at iota = pi/2 (as in SNRtsCompute)."""
    p = dict(lik.base, Mc=Mc, eta=eta, chi1z=c1, chi2z=c2, iota=np.pi/2)
    S = np.interp(lik.f, lik.est.PSDs['L1'][:, 0], lik.est.PSDs['L1'][:, 1], left=1., right=1.)
    o = {}
    for l, m in ((2, 2), (3, 3), (4, 4)):
        hp = lik.W.hphc(lik.f, **dict(p, modes=[[l, m], [l, -m]]))[0]
        o[l] = np.sum(np.abs(hp)**2/S)
    return np.sqrt(o[3]/o[2]), np.sqrt(o[4]/o[2])


def work(iev):
    import likelihood_cloud as lc
    d, rho_mf = data()
    truth = {k: float(v[iev]) for k, v in d.truth.items()}
    ml = {k: float(v) for k, v in d.filter_params(iev).items()}
    out = np.full(6, np.nan)
    try:
        lik = lc.DataLikelihood(truth, int(d.idxs_det[iev]), ml)
        rho_ml = lik.rho(*(ml[k] for k in INTR))
        rho_tr = lik.rho(*(truth[k] for k in INTR))
        r33, r44 = mode_snr_ratios(lik, *(truth[k] for k in INTR))
        out[:6] = [r33, r44, rho_ml, rho_tr, *mode_snr_ratios(lik, *(ml[k] for k in INTR))]
    except Exception as exc:     # LAL domain errors
        print(iev, repr(exc), flush=True)
    return out


if __name__ == '__main__':
    n_procs = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    d, rho_mf = data()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else d.n_events
    t0 = time.time()
    res = np.zeros((n, 6))
    with Pool(n_procs, maxtasksperchild=100) as pool:
        for k, r in enumerate(pool.imap(work, range(n), chunksize=8), 1):
            res[k - 1] = r
            if k % 500 == 0:
                print(f'[{time.time() - t0:6.0f} s] {k}/{n}', flush=True)
    ml = {k: np.asarray(d.ml[k][:n], float) for k in list(INTR) + ['dL', 'iota', 'ML_fallback', 'ML_twoDeltalnL']}
    tr = {k: np.asarray(d.truth[k][:n], float) for k in list(INTR) + ['dL', 'iota', 'm2_src', 'm2_det', 'z'] if k in d.truth}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, rho_mf=rho_mf[:n], R33=res[:, 0], R44=res[:, 1], rho_ml_profile=res[:, 2],
             rho_true_profile=res[:, 3], R33_ml=res[:, 4], R44_ml=res[:, 5],
             **{f'ml_{k}': v for k, v in ml.items()}, **{f'true_{k}': v for k, v in tr.items()})
    print('wrote', OUT, f'({time.time() - t0:.0f} s)')
