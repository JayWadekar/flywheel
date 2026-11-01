"""Per-event access to the O5 NSBH data sets and the cogwheel extrinsic marginalization.

Data sets (under config.data_root(), made by SNR_timeseries/):
  'truth' : zero-noise time series filtered with the true template, keyed by row 0..16501
  'ML'    : seed-42 Gaussian noise, filtered with the maximum-likelihood (ML) template fitted to the noisy data
            (SNRtsML.py), keyed by catalog index (row iev <-> idxs_det[iev]); the ML parameters are in
            filter_params_*.hdf5, in the same row order
"""

import glob, os
import numpy as np, h5py, pandas as pd
import config

POP = 'NSBH_cat_1yr_LVKpop_100realizations_HMloc_L1-H1-Virgo_O5_LAL-IMRPhenomXHM_0to25023612'
TS_SUBDIRS = {'truth': f'SNRts_{POP}/', 'ML': f'SNRts_{POP}_noise42_MLtemplate/'}
DETS = ['H1', 'L1', 'V1']
MODES = ['22', '33', '44']


def det_key(det):
    return 'Virgo' if det == 'V1' else det


class O5Data:
    """Lazy access to one of the O5 data sets."""

    def __init__(self, kind='ML'):
        self.kind = kind
        d = config.data_root() + TS_SUBDIRS[kind]
        self.ts = h5py.File(glob.glob(d + 'snrs_timeseries_*.hdf5')[0], 'r')
        self.opt = h5py.File(glob.glob(d + 'snrs_opt_info_*.hdf5')[0], 'r')
        with h5py.File(glob.glob(d + 'detected_events_*.hdf5')[0], 'r') as f:
            self.truth = {k: f[k][:] for k in f}
        self.idxs_det = np.loadtxt(glob.glob(d + 'idxs_det_*.txt')[0]).astype(int)
        self.ml = None
        if kind == 'ML':
            with h5py.File(glob.glob(d + 'filter_params_*.hdf5')[0], 'r') as f:
                self.ml = {k: f[k][:] for k in f}
            assert np.array_equal(self.ml['catalog_idx'].astype(int), self.idxs_det)
        self.n_events = len(self.idxs_det)

    def key(self, iev):
        return str(iev) if self.kind == 'truth' else str(self.idxs_det[iev])

    def event(self, iev):
        """(times, SNR_ts[det][mode], SNR_sq_opt[det][mode] incl. 'modes_correlation') for row iev."""
        g, o = self.ts[self.key(iev)], self.opt[self.key(iev)]
        snr_ts = {det: {m: g[det][m][()] for m in g[det]} for det in g if isinstance(g[det], h5py.Group)}
        snr_opt = {det: {m: o[det][m][()][0] for m in o[det]} for det in o}
        return g['tgrid'][()], snr_ts, snr_opt

    def filter_params(self, iev):
        """Parameters of the template used to filter row iev (ML for 'ML', truth for 'truth')."""
        src = self.ml if self.kind == 'ML' else self.truth
        return {k: v[iev] for k, v in src.items()}


def build_inputs(snr_ts, snr_opt, dist_factor_ref, incoherent_half=False):
    """dh_mtd, hh_md, incoherent_lnprob_td for cogwheel (L1 is the reference detector), as in the example notebook.
    The incoherent arrival-time proposal only affects the sampling efficiency, not the posterior."""
    dh_mtd = []
    for mode in MODES:
        dh_mtd.append(np.c_[[snr_ts[det_key(d)][mode] * (1 if d == 'L1' else
                             np.sqrt(snr_opt[det_key(d)][mode]/snr_opt['L1'][mode])) for d in DETS]].T)
    dh_mtd = np.array(dh_mtd)*dist_factor_ref
    pairs = ['22', '33', '44', '33', '44', '44']
    hh_md = np.array([[snr_opt[det_key(d)][m]/snr_opt['L1'][m] for d in DETS] for m in pairs]).astype(complex)
    hh_md *= dist_factor_ref**2
    corr = snr_opt['modes_correlation']
    hh_md = (hh_md.T*np.array([1, 2*corr['22-33'], 2*corr['22-44'], 1, 2*corr['33-44'], 1])).T
    inc = np.array([sum(np.abs(snr_ts[det_key(d)][m])**2 for m in MODES) for d in DETS]).T
    return dh_mtd, hh_md, inc/2 if incoherent_half else inc


def make_cs_instance(seed=None):
    import coherent_score as cs
    # min_n_effective 200 with up to 2^17 QMC points: with the notebook's 50 / 2^16, the finite importance sampling
    # under-weights the far-distance tail and biases d_L by ~+0.02 sigma in the P-P test
    LOG2N_QMC, NPHI = 12, 512
    MAX_LOG2N_QMC = int(os.environ.get('PP_MAX_LOG2N_QMC', 17))
    MIN_N_EFFECTIVE = int(os.environ.get('PP_MIN_NEFF', 200))
    inst = cs.SearchCoherentScoreHMAS(
        sky_dict=cs.cogwheel.likelihood.marginalization.SkyDictionary(''.join(d[0] for d in DETS)),
        log2n_qmc=LOG2N_QMC, nphi=NPHI, max_log2n_qmc=MAX_LOG2N_QMC, n_qmc_sequences=100, seed=seed)
    inst.min_n_effective = MIN_N_EFFECTIVE
    for i in range(len(inst._qmc_sequences)):
        inst._rng.random()
        inst._qmc_sequences[i] = inst._create_qmc_sequence()
    inst.MAX_LOG2N_QMC = MAX_LOG2N_QMC
    return inst


def gen_samples_from_marg_info(cs_instance, marg_info, num):
    if marg_info.q_inds.size == 0:
        return None
    ids = cs_instance._rng.choice(len(marg_info.q_inds), size=num, p=marg_info.weights)
    q_ids, o_ids, sky_ids = marg_info.q_inds[ids], marg_info.o_inds[ids], marg_info.sky_inds[ids]
    t_geocenter = marg_info.t_first_det[ids] - cs_instance.sky_dict.geocenter_delay_first_det[sky_ids]
    d_h, h_h = marg_info.d_h[ids], marg_info.h_h[ids]
    d_luminosity = cs_instance._sample_distance(d_h, h_h)
    ratio = d_luminosity/cs_instance.lookup_table.REFERENCE_DISTANCE
    return {'d_luminosity': d_luminosity,
            'dec': cs_instance.sky_dict.sky_samples['lat'][sky_ids],
            'lon': cs_instance.sky_dict.sky_samples['lon'][sky_ids],
            'psi': cs_instance._qmc_sequence['psi'][q_ids],
            't_geocenter': t_geocenter,
            'lnl': d_h/ratio - h_h/ratio**2/2,
            'n_effective': np.full(num, marg_info.n_effective),
            'cosiota': cs_instance._qmc_sequence['cosiota'][q_ids],
            'q_ids': q_ids,
            # Inner products at the reference distance, to redraw the distance with a corrected overlap
            'd_h': d_h, 'h_h_ref': h_h}


def run_marginalization(cs_instance, dh_mtd, hh_md, times, inc, mode_ratios_qm, n_runs=10, n_samples_per_run=1000):
    out = []
    for _ in range(n_runs):
        mi = cs_instance.get_marginalization_info(dh_mtd, hh_md, times, inc, mode_ratios_qm)
        s = gen_samples_from_marg_info(cs_instance, mi, n_samples_per_run)
        if s is not None:
            out.append(pd.DataFrame(s))
    return pd.concat(out, ignore_index=True) if out else None


def lon_to_ra(samples):
    import lal, cogwheel
    gmst = [lal.GreenwichMeanSiderealTime(t) for t in np.asarray(samples['t_geocenter'])]
    return cogwheel.skyloc_angles.lon_to_ra(samples['lon'], gmst)
