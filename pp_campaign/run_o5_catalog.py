"""Low-latency PE samples for every event of the O5 NSBH catalog (100 realizations, 16502 events) on the noisy
('ML') data set, with the Fisher + refinement cloud. Same per-event pipeline as the GW190814 figure and the P-P
campaign (run_pp_event.py), looped over the catalog; the output keeps the layout of the old zero-noise file
Marg_PE_samples_..._O5-IMRPhenomXHM.hdf5 (groups samples_HM / samples_22, one row of N_RUNS x 1000 samples per
event), plus the population weights and per-event metadata.

Per event (row iev of the ML data set; the cloud and cogwheel draws are seeded with 1000 + iev):
 1. Fisher at the ML template + adaptive importance sampling of the relative-binning 22 likelihood x population
    prior at z_ML (z_cloud: nearest redshift with population support when z_ML has none) -> 1000 cloud members with R33, R44 and match m_i = rho_i/rho_ML.
 2. cogwheel HM run (R from the cloud) and 22-only run (R = 0), N_RUNS x 1000 samples each.
 3. Overlap correction: d_L redrawn from cogwheel's lookup table with m_i (d|h); lnl evaluated there.
 4. weight = p_pop(d_L)/d_L^2 x [mass prior at z(d_L)]/[mass prior at z_cloud]: the samples themselves are drawn
    under the analysis prior (d_L^2), as in the old file; use the weights for the population prior.

Datasets: samples_{HM,22}/{d_luminosity [Gpc, as the old file], iota, m2_src, psi, ra, dec, n_effective, lnl,
weight}; per event: catalog_idx, rho_mf, ML_fallback, done, cloud_ess, mean_1-match, t_total_s.
Events run in a fixed random permutation (seed 2026), so a partial run is a random subset; reruns skip done rows.

Usage: python run_o5_catalog.py OUT.hdf5 [N_PROCS] [N_EVENTS]
"""
import os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import multiprocessing as mp
import numpy as np, h5py, pandas as pd
import config
import o5_data as od

N_RUNS, N_PER_RUN = 10, 1000
KEYS = ['d_luminosity', 'iota', 'm2_src', 'psi', 'ra', 'dec', 'n_effective', 'lnl', 'weight']
META = ['catalog_idx', 'rho_mf', 'ML_fallback', 'cloud_ess', 'mean_1-match', 't_total_s', 'z_ml', 'z_cloud']
CS = None          # cogwheel instance, built in the parent and shared by the forked workers (copy on write)
DATA = None


def process(iev):
    import cogwheel.cosmology
    import run_pp_event as rp
    import likelihood_cloud as lc
    import rb_likelihood as rb
    import population
    import flywheel.fisher as fisher
    t0 = time.time()
    seed = 1000 + iev
    rng = np.random.default_rng(seed)
    np.random.seed(rng.integers(2**31))
    pop = population.PopulationPrior()
    ml = DATA.filter_params(iev)
    truth = {k: float(v[iev]) for k, v in DATA.truth.items()}
    p = fisher.from_events({k: np.atleast_1d(v) for k, v in ml.items()}, 0)
    z_ml = float(cogwheel.cosmology.z_of_d_luminosity(ml['dL']*1000))

    # 1. cloud (the truth only regenerates the noisy data behind the stored time series)
    fm = fisher.FisherHM(rp.ASDS, fmin=rp.FMIN, fmax=rp.FMAX, mode_array=[[2, 2], [2, -2]])
    F = rp.adaptive_fisher(fm, p)
    lik = lc.DataLikelihood(truth, int(DATA.idxs_det[iev]), ml)
    z_cloud = rp.cloud_redshift(ml['Mc'], z_ml, pop)   # = z_ML unless the population has no support there
    c, cinfo = lc.likelihood_cloud(rb.RBLikelihood(lik, 0.1), p, F, z_cloud, pop, rng)
    ratios = rb.mode_ratios_fast(c, ml)
    match = np.clip(c['match'], 0.5, 1.2)
    q = fisher.eta_to_q(c['eta'])
    m2_det = q/(1 + q)*c['Mc']/c['eta']**0.6

    # 2.-4. cogwheel runs, overlap correction, population weights
    od.reseed_cs_instance(CS, seed)
    dist_factor_ref = CS.lookup_table.d_luminosity_max/CS.lookup_table.REFERENCE_DISTANCE
    times, snr_ts, snr_opt = DATA.event(iev)
    dh_mtd, hh_md, inc = od.build_inputs(snr_ts, snr_opt, dist_factor_ref)
    to_gpc = np.sqrt(snr_opt['L1']['22'])*2*ml['dL']/dist_factor_ref
    ref = CS.lookup_table.REFERENCE_DISTANCE
    out = {}
    for label in ('HM', '22'):
        ids = rng.choice(len(match), size=2**CS.MAX_LOG2N_QMC)
        r = ratios[ids] if label == 'HM' else ratios[ids]*0
        s = od.run_marginalization(CS, dh_mtd, hh_md, times, inc, r, n_runs=N_RUNS, n_samples_per_run=N_PER_RUN)
        for _ in range(5):          # a run with no surviving QMC points returns nothing: add runs until N_RUNS
            n_have = 0 if s is None else len(s)
            if n_have == N_RUNS*N_PER_RUN:
                break
            extra = od.run_marginalization(CS, dh_mtd, hh_md, times, inc, r, n_runs=N_RUNS - n_have//N_PER_RUN,
                                           n_samples_per_run=N_PER_RUN)
            s = extra if s is None else (s if extra is None else pd.concat([s, extra], ignore_index=True))
        if s is None or len(s) != N_RUNS*N_PER_RUN:
            raise RuntimeError(f'{label}: cogwheel returned {0 if s is None else len(s)} samples')
        member = ids[s['q_ids'].to_numpy()] if label == 'HM' else rng.choice(len(match), size=len(s))
        dh = match[member]*s['d_h'].to_numpy()
        hh = s['h_h_ref'].to_numpy()
        d_norm = CS._sample_distance(dh, hh)
        dL = d_norm*to_gpc
        z = cogwheel.cosmology.z_of_d_luminosity(dL*1000)
        num = rp.mass_prior_lnMc_eta(c['Mc'][member], c['eta'][member], z, pop)
        den = rp.mass_prior_lnMc_eta(c['Mc'][member], c['eta'][member], z_cloud, pop)
        w = pop.dL(dL*1000)/(dL*1000)**2*np.where(den > 0, num/np.where(den > 0, den, 1), 0)
        out[label] = {'d_luminosity': dL, 'iota': np.arccos(s['cosiota'].to_numpy()),
                      'm2_src': m2_det[member]/(1 + z), 'psi': s['psi'].to_numpy(),
                      'ra': np.asarray(od.lon_to_ra(s)), 'dec': s['dec'].to_numpy(),
                      'n_effective': s['n_effective'].to_numpy(), 'lnl': dh*ref/d_norm - hh*(ref/d_norm)**2/2,
                      'weight': w/w.mean() if w.sum() > 0 else w}
    meta = {'cloud_ess': cinfo['lnl_cloud_ess'], 'z_cloud': z_cloud, 'z_ml': z_ml, 'mean_1-match': float(np.mean(1 - match)),
            't_total_s': time.time() - t0}
    return iev, out, meta


def init_worker():
    """Fresh HDF5 handles in each worker (handles opened before a fork must not be shared)."""
    global DATA
    DATA = od.O5Data('ML')


def work(iev):
    try:
        return process(iev)
    except Exception as exc:
        return iev, None, {'error': repr(exc)}


def create(fn, n):
    with h5py.File(fn, 'a') as f:
        if 'done' in f:
            return
        for g in ('samples_HM', 'samples_22'):
            grp = f.create_group(g)
            for k in KEYS:
                grp.create_dataset(k, shape=(n, N_RUNS*N_PER_RUN), dtype='float32', chunks=(1, N_RUNS*N_PER_RUN),
                                   fillvalue=np.nan)
        f.create_dataset('done', data=np.zeros(n, bool))
        rho = np.zeros(n)
        for iev in range(n):
            g = DATA.ts[DATA.key(iev)]
            rho[iev] = np.sqrt(sum(np.max(np.abs(g[d]['22'][()]))**2 for d in g if isinstance(g[d], h5py.Group)))
        f.create_dataset('catalog_idx', data=DATA.idxs_det)
        f.create_dataset('rho_mf', data=rho)
        f.create_dataset('ML_fallback', data=DATA.ml['ML_fallback'].astype(int))
        for k in ('cloud_ess', 'mean_1-match', 't_total_s', 'z_ml', 'z_cloud'):
            f.create_dataset(k, data=np.full(n, np.nan))
        f.attrs['description'] = __doc__
        f.attrs['d_luminosity_unit'] = 'Gpc'
        f.attrs['data_set'] = od.TS_SUBDIRS['ML']


if __name__ == '__main__':
    fn = sys.argv[1]
    n_procs = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    DATA = od.O5Data('ML')
    n = DATA.n_events
    create(fn, n)
    with h5py.File(fn, 'r') as f:
        done = f['done'][:]
    order = np.random.default_rng(2026).permutation(n)
    n_events = int(sys.argv[3]) if len(sys.argv) > 3 else n
    todo = [int(i) for i in order[:n_events] if not done[i]]
    DATA.ts.close(); DATA.opt.close()
    t0 = time.time()
    CS = od.make_cs_instance()
    print(f'cogwheel instance built in {time.time() - t0:.0f} s; {len(todo)} events to run on {n_procs} processes',
          flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with mp.get_context('fork').Pool(n_procs, initializer=init_worker) as pool, h5py.File(fn, 'r+') as f:
        for k, (iev, out, meta) in enumerate(pool.imap_unordered(work, todo, chunksize=1), 1):
            if out is None:
                n_fail += 1
                print(f'event {iev} failed: {meta["error"]}', flush=True)
                continue
            for label, d in out.items():
                for key, v in d.items():
                    f[f'samples_{label}'][key][iev] = v.astype(np.float32)
            for key, v in meta.items():
                if key not in f:      # files created before z_ml / z_cloud were stored
                    f.create_dataset(key, data=np.full(n, np.nan))
                f[key][iev] = v
            f['done'][iev] = True
            n_ok += 1
            if k % 100 == 0:
                f.flush()
                el = time.time() - t0
                print(f'[{el/3600:6.2f} h] {k}/{len(todo)} ({n_fail} failed), {el/k:.2f} s/event wall, '
                      f'ETA {(len(todo) - k)*el/k/3600:.1f} h', flush=True)
    print(f'finished: {n_ok} done, {n_fail} failed, {(time.time() - t0)/3600:.2f} h', flush=True)
