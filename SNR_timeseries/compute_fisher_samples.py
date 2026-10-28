"""Draw intrinsic-parameter samples from a Fisher matrix and compute their mode SNRs.

For every event in a catalog this script:

1. computes the network Fisher matrix (``flywheel.fisher``) at the event's
   parameters (true or maximum-likelihood, whichever the catalog holds),
2. draws ``--n_samples`` (Mc, eta, chi1z, chi2z) samples from the marginal
   intrinsic covariance, optionally times an aligned-spin prior,
3. computes the 22/33/44 optimal SNRs of each sample at iota = pi/2 with
   ``SNRtsCompute.modeSNR_ratio`` (same convention as
   ``compute_mode_snr_ratios_from_catalog.py``).

Outputs in ``--fout`` (the layout expected by ``flywheel.load_intrinsic_samples``
and ``flywheel.load_mode_ratio_samples``)::

    sampled_<name>.h5                        Mc, eta, chi1z, chi2z sampled; other params repeated per sample
    snrs_{22,33,44}_<ref>_0_to_<N>.txt       <ref> = reference detector, N = total number of samples
    snrs_{22,33,44}_net_0_to_<N>.txt

Example (one GW190814-like O5 event, HM Fisher matrix, bilby's AlignedSpin a <= 0.6 prior)::

    python compute_fisher_samples.py --fname_obs ../data/example/snr_timeseries/detected_events_0_to_1.hdf5 \\
        --fout ../data/fisher_samples --net L1 H1 Virgo \\
        --psds AplusDesign.txt AplusDesign.txt avirgo_O5low_NEW.txt --fmin_fisher 10 --fmax_fisher 4096
"""

import argparse, copy, json, os, sys, time
import numpy as np, h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    import flywheel.fisher as fisher
except ImportError:
    sys.path.insert(0, os.path.join(HERE, '..', 'src'))
    import flywheel.fisher as fisher
import SNRtsGlobals as glob
import SNRtsCompute as tsCompute
import SNRtsUtils as utils
import waveforms as wf

SAMPLE_KEYS = ['Mc', 'eta', 'chi1z', 'chi2z']
BILBY_NAMES = {'Virgo': 'V1'}


def event_fixed_params(ev):
    """Non-sampled parameters in the sampled_*.h5 schema (gwfast conventions, dL in Gpc)."""
    import cogwheel.cosmology  # noqa: PLC0415
    dec = ev['dec'] if 'dec' in ev else 0.5*np.pi - ev['theta']
    ra = ev['ra'] if 'ra' in ev else ev['phi']
    return {'dL': ev['dL'], 'iota': ev['iota'], 'psi': ev['psi'], 'tGPS': ev['tGPS'], 'Phicoal': ev['Phicoal'],
            'theta': 0.5*np.pi - dec, 'phi': ra,
            'tcoal': ev['tcoal'] if 'tcoal' in ev else utils.GPSt_to_LMST(ev['tGPS'], 0., 0.),
            'z': ev['z'] if 'z' in ev else float(cogwheel.cosmology.z_of_d_luminosity(ev['dL']*1000))}


def main(FLAGS):
    psd_paths = [p if os.path.isabs(p) or os.path.exists(p) else os.path.join(glob.detPath, p) for p in FLAGS.psds]
    asd_files = {BILBY_NAMES.get(d, d): p for d, p in zip(FLAGS.net, psd_paths)}
    mode_array = None if FLAGS.fisher_modes == 'HM' else [[2, 2], [2, -2]]
    fm = fisher.FisherHM(asd_files, fmin=FLAGS.fmin_fisher, fmax=FLAGS.fmax_fisher, df=FLAGS.df,
                         mode_array=mode_array)
    spin_prior = fisher.aligned_spin_prior(FLAGS.spin_amax) if FLAGS.spin_amax > 0 else None

    net = {k: copy.deepcopy(glob.detectors).pop(k) for k in FLAGS.net}
    for det, p in zip(FLAGS.net, psd_paths):
        net[det]['psd_path'] = p
    wf_model = wf.LAL_WF(FLAGS.wf_model.split('-')[1], is_HigherModes=True)
    if hasattr(wf_model, 'fRef'):
        wf_model.fRef = FLAGS.fmin_ratios
    snr_simulator = tsCompute.modeSNR_ratio(net, is_ASD=FLAGS.is_ASD, fmin=FLAGS.fmin_ratios, fmax=FLAGS.fmax_ratios,
                                            reference_detector=FLAGS.reference_detector,
                                            individual_modes=FLAGS.modes_list)

    events = utils.load_population(FLAGS.fname_obs)
    n_events = len(events[list(events)[0]])
    idx_f = n_events if FLAGS.idx_f is None else FLAGS.idx_f
    rng = np.random.default_rng(FLAGS.seed)

    out = {k: [] for k in SAMPLE_KEYS + ['dL', 'iota', 'psi', 'tGPS', 'Phicoal', 'theta', 'phi', 'tcoal', 'z']}
    snrs = {(m, d): [] for m in FLAGS.modes_list for d in (FLAGS.reference_detector, 'net')}
    info = []
    for iev in range(FLAGS.idx_in, idx_f):
        ti = time.time()
        ev = {k: np.atleast_1d(v)[iev] for k, v in events.items()}
        p = fisher.from_events(events, iev)
        cov = fisher.intrinsic_covariance(fm.fisher(p), 'marginal')
        cloud = fisher.sample_cloud(p, cov, FLAGS.n_samples, spin_prior=spin_prior, rng=rng)
        fixed = event_fixed_params(ev)
        for k in SAMPLE_KEYS:
            out[k].append(cloud[k])
        for k, v in fixed.items():
            out[k].append(np.full(FLAGS.n_samples, v))
        for i in range(FLAGS.n_samples):
            tmpev = {**{k: cloud[k][i] for k in SAMPLE_KEYS}, **fixed,
                     'ra': fixed['phi'], 'dec': 0.5*np.pi - fixed['theta']}
            res = snr_simulator(tmpev, wf_model, df=FLAGS.df)
            for m in FLAGS.modes_list:
                snrs[(m, FLAGS.reference_detector)].append(np.sqrt(res[FLAGS.reference_detector][m]))
                snrs[(m, 'net')].append(np.sqrt(res['net'][m]))
        q = fisher.eta_to_q(cloud['eta'])
        m2det = q/(1+q)*cloud['Mc']/cloud['eta']**0.6
        info.append({'event': iev, 'sd_intrinsic_gaussian': list(np.sqrt(np.diag(cov))), 'sd_m2_det': float(m2det.std())})
        print(f'Event {iev}: sd(m2_det)={m2det.std():.4f}, sd(q)={q.std():.4f}, '
              f'sd(chi1z)={cloud["chi1z"].std():.3f}, sd(chi2z)={cloud["chi2z"].std():.3f} '
              f'({time.time() - ti:.1f} s)')

    os.makedirs(FLAGS.fout, exist_ok=True)
    n_tot = (idx_f - FLAGS.idx_in)*FLAGS.n_samples
    name = FLAGS.name or os.path.splitext(os.path.basename(FLAGS.fname_obs))[0]
    meta = {**vars(FLAGS), 'events': info}
    with h5py.File(os.path.join(FLAGS.fout, f'sampled_{name}.h5'), 'w') as f:
        for k, v in out.items():
            f.create_dataset(k, data=np.concatenate(v))
        f.attrs['meta'] = json.dumps(meta)
    for (m, d), v in snrs.items():
        np.savetxt(os.path.join(FLAGS.fout, f'snrs_{m}_{d}_0_to_{n_tot}.txt'), np.array(v))
    print(f'Wrote {n_tot} samples ({idx_f - FLAGS.idx_in} events x {FLAGS.n_samples}) to {FLAGS.fout}')


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--fname_obs", type=str, required=True, help='Catalog (h5) with Mc (detector frame), eta, chi1z, chi2z, dL [Gpc], iota, psi, ra/dec or phi/theta, tGPS, Phicoal: true or ML parameters.')
parser.add_argument("--fout", type=str, required=True, help='Output folder (created if missing).')
parser.add_argument("--name", default=None, type=str, help='Name used in sampled_<name>.h5 (default: catalog file name).')
parser.add_argument("--idx_in", default=0, type=int, help='First catalog index.')
parser.add_argument("--idx_f", default=None, type=int, help='One past the last catalog index (default: all).')
parser.add_argument("--n_samples", default=1000, type=int, help='Samples per event (flywheel loaders default to 1000).')
parser.add_argument("--net", nargs='+', default=['L1', 'H1', 'Virgo'], type=str, help='Detectors (SNRtsGlobals names).')
parser.add_argument("--psds", nargs='+', required=True, type=str, help='ASD/PSD file of each detector: a path, or a file name inside psds/.')
parser.add_argument("--is_ASD", default=True, type=bool, help='PSD files contain ASDs (the Fisher matrix always reads them as ASDs).')
parser.add_argument("--fisher_modes", default='HM', choices=['HM', '22'], help='Modes in the Fisher matrix: HM for the 22+HM run, 22 for the 22-only run.')
parser.add_argument("--fmin_fisher", default=10., type=float, help='Minimum frequency of the Fisher matrix (Hz); match the analysis you compare with.')
parser.add_argument("--fmax_fisher", default=4096., type=float, help='Maximum frequency of the Fisher matrix (Hz).')
parser.add_argument("--fmin_ratios", default=10., type=float, help='Minimum frequency of the mode SNRs (match the SNR time series).')
parser.add_argument("--fmax_ratios", default=4096., type=float, help='Maximum frequency of the mode SNRs.')
parser.add_argument("--df", default=1/8., type=float, help='Frequency resolution (Hz).')
parser.add_argument("--spin_amax", default=0.6, type=float, help='AlignedSpin prior a_max applied to chi1z and chi2z; <= 0 for none.')
parser.add_argument("--wf_model", default='LAL-IMRPhenomXHM', type=str, help='Waveform for the mode SNRs (the Fisher matrix uses IMRPhenomXHM).')
parser.add_argument("--modes_list", nargs='+', default=['22', '33', '44'], type=str)
parser.add_argument("--reference_detector", default='L1', type=str)
parser.add_argument("--seed", default=0, type=int)

if __name__ == '__main__':
    main(parser.parse_args())
