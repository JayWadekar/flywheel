"""One event of the O5 P-P campaign: low-latency PE on noisy data filtered with the ML (search-like) template.

Per event:
 1. Intrinsic cloud of 1000 members (Mc, eta, chi1z, chi2z), with R33, R44 and the match m = rho_i/rho_ML of each:
    PP_CLOUD=rb   (default) Fisher matrix at the ML template, adaptive importance sampling of the relative-binning
                  22-mode likelihood x population prior (likelihood_cloud.py, rb_likelihood.py), R33/R44 on a sparse
                  frequency grid.
    PP_CLOUD=flow conditional normalizing flow trained on simulated searches (flow/train_flow.py): cloud, R33, R44
                  and match from the search output alone (no waveform calls). PP_FLOW_FILE selects the model.
 2. Cogwheel marginalization over the extrinsic parameters: HM run with the cloud's R33/R44, 22-only run with R = 0.
    Each posterior sample carries the cloud member whose R it used (HM) or a random member (22-only).
 3. Distance overlap correction: d_L is redrawn from cogwheel's lookup table with m_i (d|h) instead of (d|h), where
    (d|h) comes from the ML template's time series. Groups 'HM', '22' hold the corrected posteriors, 'HM_raw',
    '22_raw' the uncorrected ones.
 4. Population weights: p_pop(dL)/dL^2 x [mass prior moved from z_ML to z(dL_i)].
 5. Weighted credible level (fraction of posterior samples below the truth) of dL, iota, ra, dec, m2_src.

Usage: python run_pp_event.py iev [out_dir]
"""
import sys, os, time, json
import numpy as np, h5py
import config
import o5_data as od
import population
import flywheel.fisher as fisher

ASDS = config.ASDS
FMIN, FMAX = 10., 4096.          # same as the SNR time series
KAPPA, SPIN_REG = 0.3, 0.5       # Fisher step = KAPPA/sqrt(F_ii); N(0, SPIN_REG) spin term in the first proposal
N_CLOUD = 1000
PP_KEYS = ['d_luminosity', 'iota', 'ra', 'dec', 'm2_src']
CLOUD = os.environ.get('PP_CLOUD', 'rb')
FD_PARAMS = [k for k in fisher.PARAMS if k not in ('lndL', 'tc')]


def adaptive_fisher(fm, p, kappa=KAPPA, n_iter=2):
    """Fisher matrix with steps adapted to the event: step_i = kappa/sqrt(F_ii), iterated n_iter times."""
    s = (np.exp(p['lnMc'])/6.42)**(5/3)
    steps = {k: v*(s if k in fisher.INTRINSIC else 1) for k, v in fisher.DEFAULT_STEPS.items()}
    for _ in range(n_iter + 1):
        F = fm.fisher(p, steps=steps)
        steps = {k: kappa/np.sqrt(F[fisher.PARAMS.index(k), fisher.PARAMS.index(k)]) for k in FD_PARAMS}
    return F


def masses_src(Mc_det, eta, z):
    q = fisher.eta_to_q(eta)
    M = Mc_det/eta**0.6/(1 + z)
    return M/(1 + q), q*M/(1 + q)


def mass_prior_lnMc_eta(Mc_det, eta, z, pop):
    """Population prior density in (ln Mc_det, eta) at redshift z (up to a constant):
    p(m1s, m2s) |d(m1s, m2s)/d(lnMc_det, eta)| = p(m1s, m2s) M_det^2 / (sqrt(1-4 eta) (1+z)^2)."""
    m1, m2 = masses_src(Mc_det, eta, z)
    M_det = Mc_det/eta**0.6
    return pop.m12(m1, m2)*M_det**2/np.sqrt(np.clip(1 - 4*eta, 1e-12, None))/(1 + z)**2


def credible_level(x, w, truth):
    return float(np.sum(w*(x < truth))/np.sum(w))


def process(iev, out_dir, seed=None):
    import cogwheel.cosmology
    t0 = time.time()
    rng = np.random.default_rng(seed if seed is not None else 1000 + iev)
    np.random.seed(rng.integers(2**31))   # cogwheel helpers use the global RNG
    pop = population.PopulationPrior()
    data = od.O5Data('ML')
    ml = data.filter_params(iev)
    truth = {k: v[iev] for k, v in data.truth.items()}
    p = fisher.from_events({k: np.atleast_1d(v) for k, v in ml.items()}, 0)
    z_ml = float(cogwheel.cosmology.z_of_d_luminosity(ml['dL']*1000))

    info = {}
    if CLOUD == 'flow':
        sys.path.insert(0, os.path.join(config.HERE, 'flow'))
        import torch
        import train_flow as tf
        torch.set_num_threads(1)
        t1 = time.time()
        flow = tf.IntrinsicFlow.load(os.path.join(tf.RES, os.environ.get('PP_FLOW_FILE', 'flow.pt')))
        T = np.load(os.path.join(tf.RES, 'training_set.npz'))
        ctx = tf.context({k: T[k][iev:iev + 1] for k in T.files})[0]
        assert np.isclose(T['ml_Mc'][iev], ml['Mc'])
        c = flow.sample(ctx, N_CLOUD, seed=int(rng.integers(2**31)))
        info['t_cloud_s'] = time.time() - t1
        ratios = np.c_[c['R33'], c['R44']]
    elif CLOUD == 'rb':
        import likelihood_cloud as lc
        import rb_likelihood as rb
        fm = fisher.FisherHM(ASDS, fmin=FMIN, fmax=FMAX, mode_array=[[2, 2], [2, -2]])
        F = adaptive_fisher(fm, p)
        info['t_fisher_s'] = time.time() - t0
        # The truth only simulates the data (same signal and noise realization as the stored time series)
        lik = lc.DataLikelihood({k: float(v) for k, v in truth.items()}, int(data.idxs_det[iev]), ml)
        t1 = time.time()
        rbl = rb.RBLikelihood(lik, 0.1)
        info['t_rb_setup_s'] = time.time() - t1
        t1 = time.time()
        c, cinfo = lc.likelihood_cloud(rbl, p, F, z_ml, pop, rng)
        info['t_cloud_s'] = time.time() - t1
        info.update(cinfo)
        t1 = time.time()
        ratios = rb.mode_ratios_fast(c, ml)
        info['t_R_s'] = time.time() - t1
    else:
        raise ValueError(f'PP_CLOUD={CLOUD}')
    cloud = {k: c[k] for k in ('Mc', 'eta', 'chi1z', 'chi2z')}
    match = np.clip(c['match'], 0.5, 1.2)
    info['mean_1-match'] = float(np.mean(1 - match))
    t_cloud = time.time() - t0

    cs_instance = od.make_cs_instance()
    dist_factor_ref = cs_instance.lookup_table.d_luminosity_max/cs_instance.lookup_table.REFERENCE_DISTANCE
    times, snr_ts, snr_opt = data.event(iev)
    dh_mtd, hh_md, inc = od.build_inputs(snr_ts, snr_opt, dist_factor_ref)
    to_mpc = np.sqrt(snr_opt['L1']['22'])*2*ml['dL']*1000/dist_factor_ref

    res = {}
    for label in ['HM', '22']:
        ids = rng.choice(N_CLOUD, size=2**cs_instance.MAX_LOG2N_QMC)
        r = ratios[ids] if label == 'HM' else ratios[ids]*0
        t1 = time.time()
        s = od.run_marginalization(cs_instance, dh_mtd, hh_md, times, inc, r)
        info[f't_cogwheel_{label}_s'] = time.time() - t1
        if s is None:
            res[label] = None
            continue
        member = ids[s['q_ids'].to_numpy()] if label == 'HM' else rng.choice(N_CLOUD, size=len(s))
        q = fisher.eta_to_q(cloud['eta'][member])
        m2_det = q/(1 + q)*cloud['Mc'][member]/cloud['eta'][member]**0.6
        tru = {'d_luminosity': truth['dL']*1000, 'iota': truth['iota'], 'ra': truth['ra'],
               'dec': truth['dec'], 'm2_src': truth['m2_src']}
        common = {'iota': np.arccos(s['cosiota'].to_numpy()), 'ra': od.lon_to_ra(s), 'dec': s['dec'].to_numpy()}
        # 'raw': distance as sampled with the ML-template overlap.
        # 'corr': distance redrawn with the overlap of each sample's own template, m_i * (d|h); the
        #         intrinsic posterior stays the cloud (no extra weight), only the amplitude-distance mapping changes.
        dL_raw = s['d_luminosity'].to_numpy()*to_mpc
        dL_corr = cs_instance._sample_distance(match[member]*s['d_h'].to_numpy(), s['h_h_ref'].to_numpy())*to_mpc
        res[label] = {'n_effective': float(np.median(s['n_effective']))}
        for tag, dL in [('raw', dL_raw), ('corr', dL_corr)]:
            z = cogwheel.cosmology.z_of_d_luminosity(dL)
            post = {**common, 'd_luminosity': dL, 'm2_src': m2_det/(1 + z)}
            # Population reweighting: distance prior, and the mass prior moved from z_ML to z(dL_i)
            num = mass_prior_lnMc_eta(cloud['Mc'][member], cloud['eta'][member], z, pop)
            den = mass_prior_lnMc_eta(cloud['Mc'][member], cloud['eta'][member], z_ml, pop)
            w = pop.dL(dL)/dL**2*np.where(den > 0, num/np.where(den > 0, den, 1), 0)
            res[label][tag] = {'post': post, 'w': w,
                               'cl': {k: credible_level(post[k], w, tru[k]) for k in PP_KEYS},
                               'cl_unweighted': {k: credible_level(post[k], np.ones_like(w), tru[k]) for k in PP_KEYS},
                               'ess_w': float(w.sum()**2/np.sum(w**2)) if w.sum() > 0 else 0.}

    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(os.path.join(out_dir, f'event_{iev:05d}.h5'), 'w') as f:
        f.attrs['iev'] = iev
        f.attrs['catalog_idx'] = int(data.idxs_det[iev])
        f.attrs['info'] = json.dumps({**info, 't_cloud_total_s': t_cloud, 't_total_s': time.time() - t0, 'z_ml': z_ml})
        f.attrs['truth'] = json.dumps({k: float(v) for k, v in truth.items()})
        for label, r in res.items():
            for tag in ('raw', 'corr'):
                g = f.create_group(label if tag == 'corr' else f'{label}_raw')
                if r is None:
                    g.attrs['failed'] = True
                    continue
                rt = r[tag]
                for k, v in rt['post'].items():
                    g.create_dataset(k, data=v.astype(np.float32))
                g.create_dataset('weight', data=rt['w'].astype(np.float32))
                g.attrs['cl'] = json.dumps(rt['cl'])
                g.attrs['cl_unweighted'] = json.dumps(rt['cl_unweighted'])
                g.attrs['ess_w'] = rt['ess_w']
                g.attrs['n_effective'] = r['n_effective']
        g = f.create_group('cloud')
        for k, v in cloud.items():
            g.create_dataset(k, data=v)
        g.create_dataset('R33_R44', data=ratios)
        g.create_dataset('match', data=match)
    return iev, time.time() - t0


if __name__ == '__main__':
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(config.RESULTS, f'pp_O5_ML_{CLOUD}')
    iev, dt = process(int(sys.argv[1]), out)
    with h5py.File(os.path.join(out, f'event_{iev:05d}.h5'), 'r') as f:
        print(f'event {iev}: {dt:.0f} s', json.loads(f.attrs['info']))
        for label in ['HM', '22']:
            if 'cl' in f[label].attrs:
                print(f'  {label}: cl={json.loads(f[label].attrs["cl"])} ess_w={f[label].attrs["ess_w"]:.0f} n_eff={f[label].attrs["n_effective"]:.0f}')
