import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import multiprocessing
# Needed for mpipool not to stall when trying to write on a file (do not ask me why)
multiprocessing.set_start_method("spawn",force=True)

PACKAGE_PARENT = '../SNR_timeseries'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd())))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR,PACKAGE_PARENT )))
import copy
import numpy as np
import argparse
import h5py

import SNRtsGlobals as glob
import waveforms as wf
import SNRtsML
from compute_snrts_from_catalog import Logger, get_pool, get_events_subset, load_population, save_events

#####################################################################################
# Maximum likelihood estimation on noisy data for the events of a catalog.
# The data of each event are the same as in compute_snrts_from_catalog.py run with --add_noise 1 and the same
# --noise_parent_seed, network, PSDs, fmin, fmax and injection waveform: the noise depends only on
# (noise_parent_seed, index of the event in the catalog, detector name), see SNRtsNoise.py.
#####################################################################################

def to_file_ML(res, out_path, suff='', verbose=True, attrs=None):
    fname_out = os.path.join(out_path, 'ML_params'+suff+'.hdf5')
    if verbose:
        print('Saving ML results to file %s' %fname_out)
    with h5py.File(fname_out, 'w') as f:
        for key, value in res.items():
            f.create_dataset(key, data=np.asarray(value), compression='gzip', shuffle=False)
        if attrs is not None:
            for key, value in attrs.items():
                f.attrs[key] = value

def from_file_ML(out_path, suff='', verbose=True):
    fname_out = os.path.join(out_path, 'ML_params'+suff+'.hdf5')
    if verbose:
        print('Loading ML results from file %s' %fname_out)
    with h5py.File(fname_out, 'r') as f:
        return {key: f[key][()] for key in f.keys()}

def get_waveform(wf_model_name, lalargs, fmin):
    is_tidal, is_prec, is_HM, is_ecc = 'tidal' in lalargs, 'precessing' in lalargs, 'HM' in lalargs, 'eccentric' in lalargs
    wf_model = wf.LAL_WF(wf_model_name.split('-')[1], is_tidal=is_tidal, is_HigherModes=is_HM, is_Precessing=is_prec, is_eccentric=is_ecc)
    if hasattr(wf_model, 'fRef'):
        wf_model.fRef = fmin
    return wf_model

#####################################################################################
# Actual computation of ML parameters
#####################################################################################

def main(idx, FLAGS):

        idx = idx-1

        Net= { k: copy.deepcopy(glob.detectors).pop(k) for k in FLAGS.net}
        for i,psd in enumerate(FLAGS.psds):
            Net[FLAGS.net[i]]['psd_path'] = psd

        wf_model = get_waveform(FLAGS.wf_model, FLAGS.lalargs, FLAGS.fmin)
        if FLAGS.wf_model_rec is None:
            wf_model_rec, wf_model_rec_name = wf_model, FLAGS.wf_model
        else:
            wf_model_rec, wf_model_rec_name = get_waveform(FLAGS.wf_model_rec, FLAGS.lalargs_rec, FLAGS.fmin), FLAGS.wf_model_rec

        logidx = '_'+str(FLAGS.idx_in)+'_to_'+str(FLAGS.idx_f)
        if FLAGS.resume_run==0:
            logfile = os.path.join(FLAGS.fout, 'logfile_ML'+logidx+'_'+str(idx)+'.txt')
        else:
            logfile = os.path.join(FLAGS.fout, 'logfile_ML'+logidx+'_'+str(idx)+'_resume_'+str(FLAGS.resume_run)+'.txt')
        myLog = Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog

        print('\n------------ Network used:  ------------\n%s' %str(Net))
        print('----------------------------------------\n')
        print('------ Injection waveform: %s, recovery waveform: %s, recovery modes: %s ------\n' %(FLAGS.wf_model, wf_model_rec_name, FLAGS.modes_rec))

        modes_rec = FLAGS.modes_rec
        if 'all' in modes_rec:
            modes_rec = None

        ml = SNRtsML.ML_estimator(Net, wf_model_rec, is_ASD=bool(FLAGS.is_ASD), fmin=FLAGS.fmin, fmax=FLAGS.fmax, modes_rec=modes_rec,
                                  spin_prior=FLAGS.spin_prior, chirpmass_prior=FLAGS.chirpmass_prior,
                                  ml_maxiter=FLAGS.ml_maxiter, ml_tol=FLAGS.ml_tol, ml_rcond=FLAGS.ml_rcond, ml_fd_step=FLAGS.ml_fd_step,
                                  ml_max_linesearch=FLAGS.ml_max_linesearch, ml_lm_lambda0=FLAGS.ml_lm_lambda0, verbose=True)

        noise_seed = FLAGS.noise_parent_seed if FLAGS.add_noise else None
        print('Noise: %s' %('Gaussian, parent seed %s' %noise_seed if FLAGS.add_noise else 'none (zero-noise data)'))

        ti_evs=  time.time()
        for (pin, pf) in FLAGS.batches[str(idx)]:
            ti_loop=  time.time()
            suffstr = '_'+str(pin)+'_to_'+str(pf)
            if FLAGS.resume_run>0 and os.path.exists(os.path.join(FLAGS.fout, 'ML_params'+suffstr+'.hdf5')):
                print('Batch already present, skipping...')
                continue

            ev_chunk = {k: FLAGS.events[k][pin:pf] for k in FLAGS.events.keys()}
            catidx_chunk = FLAGS.catalog_idxs[pin:pf]
            print('\nIn this chunk we have %s events, catalog indexes from %s to %s' %(pf-pin, catidx_chunk[0], catidx_chunk[-1]))

            res_chunk = {}
            for iev in range(pf-pin):
                tmpev = {k: ev_chunk[k][iev] for k in ev_chunk.keys()}
                res = ml(tmpev, wf_model, noise_seed=noise_seed, event_idx=catidx_chunk[iev], df=FLAGS.df)
                for k, v in res.items():
                    res_chunk.setdefault(k, []).append(v)
            res_chunk['catalog_idx'] = catidx_chunk
            to_file_ML(res_chunk, FLAGS.fout, suff=suffstr)

            print('------ Time to compute chunk: %s sec.' %(str(time.time()-ti_loop)))

        print('------ Time to compute events: %s sec.\n\n' %( str((time.time()-ti_evs))))
        print('------ Done for CPU %s. ' %(idx))
        myLog.close()

def mainMPI(i):
    # Just a workaround to make mpi work without starmap
    return main(i, FLAGS)

parser = argparse.ArgumentParser(prog = 'compute_ML_from_catalog.py', description='Executable to compute the maximum likelihood parameters on noisy data for a catalog of events, with the same data as compute_snrts_from_catalog.py, with the possibility to parallelize over multiple CPUs.')
parser.add_argument("--fname_obs", default='', type=str, required=True, help='Path to the file containing the catalog, with the extension ``h5``.')
parser.add_argument("--fname_snrs", default='', type=str, required=False, help='Name of the file containing the optimal SNRs associated with the catalog, with the extension ``txt``.')
parser.add_argument("--fout", default='test_ML', type=str, required=True, help='Path to output folder, which has to exist before the script is launched.')
parser.add_argument("--wf_model",  default='LAL-IMRPhenomXHM', type=str, required=False, help='Name of the injection waveform model.')
parser.add_argument("--lalargs", nargs='+', default=['HM'], type=str, required=False, help='Specifications of the injection waveform, separated by *single spacing*.')
parser.add_argument("--wf_model_rec",  default=None, type=str, required=False, help='Name of the recovery waveform model, if different from the injection one.')
parser.add_argument("--lalargs_rec", nargs='+', default=[], type=str, required=False, help='Specifications of the recovery waveform, separated by *single spacing*.')
parser.add_argument("--modes_rec", nargs='+', default=['all'], type=str, required=False, help='The modes used in the recovery template, e.g. ``22 33``, or ``all``.')
parser.add_argument("--batch_size", default=1, type=int, required=False, help='Number of events saved in each intermediate file.')
parser.add_argument("--npools", default=1, type=int, required=False, help='Number of parallel processes.')
parser.add_argument("--snr_th", default=12., type=float, required=False, help='Threshold value for the SNR in fname_snrs to consider the event detectable.')
parser.add_argument("--idx_in", default=0, type=int, required=False, help='Index of the event in the catalog from which to start the calculation.')
parser.add_argument("--idx_f", default=None, type=int, required=False, help='Index of the event in the catalog from which to end the calculation.')
parser.add_argument("--fmin", default=2., type=float, required=False, help='Minimum frequency of the grid, in Hz.')
parser.add_argument("--fmax", default=4096., type=float, required=False, help='Maximum frequency of the grid, in Hz.')
parser.add_argument("--df", default=1./8., type=float, required=False, help='Frequency step of the data, in Hz (1/8 in compute_snrts_from_catalog.py).')
parser.add_argument("--net", nargs='+', default=['L1', ], type=str, required=False, help='The network of detectors to be used, separated by *single spacing*.')
parser.add_argument("--psds", nargs='+', default=['ET-0000A-18.txt', ], type=str, required=False, help='The paths to PSDs of each detector in the network, separated by *single spacing*.')
parser.add_argument("--is_ASD", default=1, type=int, required=False, help='Int specifying if the files in psds are ASDs (``1``) or PSDs (``0``).')
parser.add_argument("--mpi", default=0, type=int, required=False, help='Int specifying if the code has to parallelize using multiprocessing (``0``), or using MPI (``1``), suitable for clusters.')
parser.add_argument("--resume_run", default=0, type=int, required=False, help='Int specifying whether to resume previous run, skipping the batches already computed.')
parser.add_argument("--add_noise", default=1, type=int, required=False, help='Int specifying whether the data contain Gaussian noise (``1``) or not (``0``).')
parser.add_argument("--noise_parent_seed", default=42, type=int, required=False, help='Parent seed of the noise, the same as in compute_snrts_from_catalog.py.')
parser.add_argument("--spin_prior", default=0.999, type=float, required=False, help='Maximum spin magnitude considered in the analysis.')
parser.add_argument("--chirpmass_prior", default=200., type=float, required=False, help='Maximum chirp mass considered in the analysis.')
parser.add_argument("--ml_maxiter", default=50, type=int, required=False, help='Maximum number of Levenberg-Marquardt iterations in the ML search.')
parser.add_argument("--ml_tol", default=1e-3, type=float, required=False, help='Convergence threshold on the decrease of -2lnL (predicted, or actual in two consecutive iterations).')
parser.add_argument("--ml_rcond", default=1e-12, type=float, required=False, help='Relative eigenvalue cutoff of the normalised Fisher matrix (numerically singular directions are not moved).')
parser.add_argument("--ml_fd_step", default=1e-2, type=float, required=False, help='Finite-difference steps are set to ml_fd_step/sqrt(F_pp), i.e. a template change of norm ~ml_fd_step.')
parser.add_argument("--ml_max_linesearch", default=12, type=int, required=False, help='Maximum number of damping increases per iteration.')
parser.add_argument("--ml_lm_lambda0", default=1e-3, type=float, required=False, help='Initial Levenberg-Marquardt damping (in Fisher-normalised units).')
parser.add_argument("--rerun_from", default=None, type=str, required=False, help='Path to the ML_params file of a previous run: only its events with ML_status in **--rerun_status** are recomputed (with the settings of this run), and a merged file with the same name is saved in **--fout**, which has to be a different folder. In this case **--fname_snrs**, **--snr_th**, **--idx_in** and **--idx_f** are ignored, while the catalog, network, PSDs, waveform, fmin, fmax and noise seed have to be the same as in the previous run.')
parser.add_argument("--rerun_status", nargs='+', default=[0, 2, 3], type=int, required=False, help='ML_status values of the events to recompute with **--rerun_from** (0 max iterations, 2 stalled, 3 error).')

if __name__ =='__main__':

    FLAGS = parser.parse_args()
    print('Input arguments: %s' %str(FLAGS))
    ti =  time.time()

    #####################################################################################
    # LOAD EVENTS
    #####################################################################################

    if not os.path.exists(FLAGS.fname_obs):
        raise ValueError('Path to catalog does not exist. Value entered: %s' %FLAGS.fname_obs)
    print('Loading events from %s...' %FLAGS.fname_obs)
    events_loaded = load_population(FLAGS.fname_obs)
    keylist = list(events_loaded.keys())
    nevents_cat = len(events_loaded[keylist[0]])
    print('This catalog has %s events.' %nevents_cat)

    if FLAGS.rerun_from is not None:
        # Rerun only the events of a previous run with the chosen ML_status values. Their data are the same as in
        # the previous run, since the noise of each event is fixed by its catalog index
        if os.path.abspath(os.path.dirname(FLAGS.rerun_from)) == os.path.abspath(FLAGS.fout):
            raise ValueError('With --rerun_from, --fout has to be a different folder from the one of the previous run.')
        rerun_name = os.path.basename(FLAGS.rerun_from)
        if not (rerun_name.startswith('ML_params') and rerun_name.endswith('.hdf5')):
            raise ValueError('--rerun_from has to be an ML_params*.hdf5 file produced by this script.')
        print('Loading previous results from %s...' %FLAGS.rerun_from)
        with h5py.File(FLAGS.rerun_from, 'r') as f:
            res_prev = {key: f[key][()] for key in f.keys()}
        for st in np.unique(res_prev['ML_status']):
            print('ML_status %d: %d events' %(st, np.sum(res_prev['ML_status']==st)))
        catalog_idxs = res_prev['catalog_idx'][np.isin(res_prev['ML_status'], FLAGS.rerun_status)].astype(int)
        nevents = len(catalog_idxs)
        print('Recomputing %s events with ML_status in %s.' %(nevents, FLAGS.rerun_status))
        if nevents == 0:
            print('Nothing to rerun.')
            sys.exit(0)
        selected = np.zeros(nevents_cat, dtype=bool)
        selected[catalog_idxs] = True
        events = get_events_subset(events_loaded, selected)
        del events_loaded
    else:
        if FLAGS.fname_snrs != '':
            snrs_loaded = np.ravel(np.loadtxt(FLAGS.fname_snrs))
            assert len(snrs_loaded)==nevents_cat, 'The SNR file has to contain one SNR per event of the catalog.'
        else:
            snrs_loaded = np.full(nevents_cat, FLAGS.snr_th + 1)

        idx_f = nevents_cat if FLAGS.idx_f is None else FLAGS.idx_f
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:idx_f] for k in events_loaded.keys()}
        snrs_loaded = snrs_loaded[FLAGS.idx_in:idx_f]
        print('Using events between %s and %s, total %s events' %(FLAGS.idx_in, idx_f, len(snrs_loaded)))

        detected = snrs_loaded > FLAGS.snr_th
        events = get_events_subset(events_loaded, detected)
        # Index of each selected event in the original catalog, which fixes its noise
        catalog_idxs = np.arange(FLAGS.idx_in, idx_f)[detected]
        nevents = len(catalog_idxs)
        print('Detected events with SNR > %s: %s' %(FLAGS.snr_th, nevents))

    if not 'ra' in events.keys():
        events['ra']  = events['phi']
        events['dec'] = 0.5*np.pi - events['theta']
        _ = events.pop('theta', None)
        _ = events.pop('phi', None)

    #####################################################################################
    # SPLIT EVENTS IN BATCHES, ASSIGNED ROUND-ROBIN TO THE PROCESSES
    # (the results do not depend on the batching, since the noise of each event is fixed by its catalog index)
    #####################################################################################

    edges = list(range(0, nevents, FLAGS.batch_size)) + [nevents]
    batches = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    if len(batches) < FLAGS.npools:
        print('Only %s batches, using %s processes instead of %s.' %(len(batches), len(batches), FLAGS.npools))
        FLAGS.npools = len(batches)
    FLAGS.batches = {str(p): batches[p::FLAGS.npools] for p in range(FLAGS.npools)}
    FLAGS.events = events
    FLAGS.catalog_idxs = catalog_idxs
    print('%s batches of max %s events on %s processes.' %(len(batches), FLAGS.batch_size, FLAGS.npools))

    ############################################################################
    # Run processes in parallel
    ############################################################################

    if FLAGS.npools>1:
        print('Parallelizing on %s CPUs ' %FLAGS.npools)
        print('Total available CPUs: %s' %str(multiprocessing.cpu_count()) )
        pool =  get_pool(mpi=FLAGS.mpi, threads=FLAGS.npools+1)
        if FLAGS.mpi:
            pool.map( mainMPI, [ i for i in range(1, FLAGS.npools+1)] )
        else:
            pool.starmap( main, [ ( i, FLAGS ) for i in range(1, FLAGS.npools+1)] )
        pool.close()
    else:
        main(1, FLAGS)

    ############################################################################
    # Concatenate results and clean
    ############################################################################

    print('\nSaving final version to file...')
    res_all = {}
    for (pin, pf) in batches:
        tmp = from_file_ML(FLAGS.fout, suff='_'+str(pin)+'_to_'+str(pf))
        for k, v in tmp.items():
            res_all.setdefault(k, []).append(v)
    res_all = {k: np.concatenate(v) for k, v in res_all.items()}

    # Settings that fix the data (signal and noise) of each event, stored as attributes of the output so that
    # other codes (compute_snrts_from_catalog.py with --fname_ML) can check that they use the same data
    run_attrs = {'wf_model': FLAGS.wf_model, 'lalargs': list(FLAGS.lalargs), 'fmin': FLAGS.fmin, 'fmax': FLAGS.fmax, 'df': FLAGS.df,
                 'net': list(FLAGS.net), 'psds': [os.path.basename(p) for p in FLAGS.psds], 'add_noise': FLAGS.add_noise,
                 'noise_parent_seed': FLAGS.noise_parent_seed,
                 'wf_model_rec': FLAGS.wf_model if FLAGS.wf_model_rec is None else FLAGS.wf_model_rec,
                 'lalargs_rec': list(FLAGS.lalargs) if FLAGS.wf_model_rec is None else list(FLAGS.lalargs_rec), 'modes_rec': list(FLAGS.modes_rec)}

    if FLAGS.rerun_from is not None:
        # Merge: the rows of the recomputed events are replaced, ML_rerun = 1 marks them
        suffstr = rerun_name[len('ML_params'):-len('.hdf5')]
        rows = {int(c): i for i, c in enumerate(res_prev['catalog_idx'])}
        pos = np.array([rows[int(c)] for c in res_all['catalog_idx']])
        merged = {k: np.array(v, dtype=float if k not in ('catalog_idx',) else v.dtype) for k, v in res_prev.items()}
        for k, v in res_all.items():
            if k not in merged:
                merged[k] = np.full(len(res_prev['catalog_idx']), np.nan)
            merged[k][pos] = v
        if 'ML_rerun' not in merged:
            merged['ML_rerun'] = np.zeros(len(res_prev['catalog_idx']))
        merged['ML_rerun'][pos] = 1.
        attrs = dict(run_attrs)
        attrs.update({'rerun_from': os.path.abspath(FLAGS.rerun_from), 'rerun_status': np.asarray(FLAGS.rerun_status)})
        for k in ('ml_maxiter', 'ml_tol', 'ml_rcond', 'ml_fd_step', 'ml_max_linesearch', 'ml_lm_lambda0'):
            attrs['rerun_'+k] = getattr(FLAGS, k)
        to_file_ML(merged, FLAGS.fout, suff=suffstr, attrs=attrs)
        to_file_ML(res_all, FLAGS.fout, suff='_rerun_only'+suffstr, attrs=attrs)
        res_all = merged
    else:
        suffstr = '_'+str(FLAGS.idx_in)+'_to_'+str(idx_f)
        to_file_ML(res_all, FLAGS.fout, suff=suffstr, attrs=run_attrs)

        print('Saving catalog of detected events...')
        events['snr'] = snrs_loaded[detected]
        events['catalog_idx'] = catalog_idxs
        save_events(os.path.join(FLAGS.fout, 'detected_events_ML'+suffstr+'.hdf5'), events)

    if len(batches)>1 or FLAGS.rerun_from is not None:
        print('Cleaning...')
        for (pin, pf) in batches:
            if '_'+str(pin)+'_to_'+str(pf) != suffstr:
                os.remove(os.path.join(FLAGS.fout, 'ML_params_'+str(pin)+'_to_'+str(pf)+'.hdf5'))

    nconv = int(np.sum(res_all['ML_status']==SNRtsML.ML_STATUS['converged']))
    print('ML search converged for %s/%s events, the others fell back to the injection parameters.' %(nconv, len(res_all['ML_status'])))
    print('Done! Total time: %s sec.' %(str(time.time()-ti)))
