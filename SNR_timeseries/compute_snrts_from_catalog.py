
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys
import time
import multiprocessing
import subprocess
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
import SNRtsUtils as utils
import waveforms as wf
import SNRtsCompute as tsCompute

#####################################################################################
# input/output logic
#####################################################################################

# Writes output both on std output and on log file
class Logger(object):
    
    def __init__(self, fname):
        self.terminal = sys.__stdout__
        self.log = open(fname, "w+")
        self.log.write('--------- LOG FILE ---------\n')
        print('Logger created log file: %s' %fname)
        #self.write('Logger')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

    def close(self):
        self.log.close
        sys.stdout = sys.__stdout__
        
    def isatty(self):
        return False

def get_pool(mpi=False, threads=None):
    """ Always returns a pool object with a `map()` method. By default,
        returns a `SerialPool()` -- `SerialPool.map()` just calls the built-in
        Python function `map()`. If `mpi=True`, will attempt to import the 
        `MPIPool` implementation from `emcee`. If `threads` is set to a 
        number > 1, it will return a Python multiprocessing pool.
        Parameters
        ----------
        mpi : bool (optional)
            Use MPI or not. If specified, ignores the threads kwarg.
        threads : int (optional)
            If mpi is False and threads is specified, use a Python
            multiprocessing pool with the specified number of threads.
    """

    if mpi:
        from schwimmbad import MPIPool
        print('Using MPI...')
        # Initialize the MPI pool
        pool = MPIPool()

        # Make sure the thread we're running on is the master
        if not pool.is_master():
            pool.wait()
            sys.exit(0)

    elif threads > 1:
        print('Using multiprocessing with %s processes...' %(threads-1))
        pool = multiprocessing.Pool(threads-1)

    else:
        raise ValueError('Called get pool with threads=1. No need')

    return pool

def get_indexes(p, all_n_per_pool):
    
    if p==0:
        pin = 0
        pf = all_n_per_pool[p]
    else: 
        pin = sum(all_n_per_pool[:p])
        pf = pin+all_n_per_pool[p]
    return pin, pf

def to_file_snr_timeseries(SNR_timeseries, out_path, suff='', verbose=True):
    fname_out = os.path.join(out_path, 'snrs_timeseries'+suff+'.hdf5')
    if verbose:
        print('Saving all snr timeseries to files. Names of file: %s' %(fname_out))
    
    with h5py.File(fname_out, 'w') as f:
        for key, value in SNR_timeseries.items():
            pg = f.create_group(str(key))
            for subkey, subvalue in value.items():
                if isinstance(subvalue, dict):
                    sub_pg = pg.create_group(subkey)
                    for subsubkey, subsubvalue in subvalue.items():
                        sub_pg.create_dataset(subsubkey, data=subsubvalue, compression='gzip', shuffle=False)
                else:
                    pg.create_dataset(subkey, data=subvalue, compression='gzip', shuffle=False)

def from_file_snr_timeseries(out_path, suff='', verbose=True):
    fname_out = os.path.join(out_path, 'snrs_timeseries'+suff+'.hdf5')
    if verbose:
        print('Loading all snr timeseries from files. Names of file: %s' %(fname_out))

    with h5py.File(fname_out, 'r') as f:
        ts_batch_loaded = {}
        for key in f.keys():
            ts_batch_loaded[int(key)] = {}
            pg = f[key]
            for subkey in pg.keys():
                if isinstance(pg[subkey], h5py.Group):
                    ts_batch_loaded[int(key)][subkey] = {subsubkey: pg[subkey][subsubkey][()] for subsubkey in pg[subkey].keys()}
                else:
                    ts_batch_loaded[int(key)][subkey] = pg[subkey][()]
                    
    return ts_batch_loaded

def to_file_snr_opt_info(SNR_opt_info, out_path, suff='', verbose=True):
    fname_out = os.path.join(out_path, 'snrs_opt_info'+suff+'.hdf5')
    if verbose:
        print('Saving all snr additional info to files. Names of file: %s' %(fname_out))
    
    with h5py.File(fname_out, 'w') as f:
        for key, value in SNR_opt_info.items():
            pg = f.create_group(str(key))
            for subkey, subvalue in value.items():
                sub_pg = pg.create_group(subkey)
                for subsubkey, subsubvalue in subvalue.items():
                    if isinstance(subsubvalue, dict):
                        subsub_pg = sub_pg.create_group(subsubkey)
                        for subsubsubkey, subsubsubvalue in subsubvalue.items():
                            subsub_pg.create_dataset(subsubsubkey, data=[subsubsubvalue], compression='gzip', shuffle=False)
                    else:
                        sub_pg.create_dataset(subsubkey, data=[subsubvalue], compression='gzip', shuffle=False)

def from_file_snr_opt_info(out_path, suff='', verbose=True):
    fname_out = os.path.join(out_path, 'snrs_opt_info'+suff+'.hdf5')
    if verbose:
        print('Loading all snr additional info from files. Names of file: %s' %(fname_out))

    with h5py.File(fname_out, 'r') as f:
        SNR_opt_info_loaded = {}
        for key in f.keys():
            SNR_opt_info_loaded[int(key)] = {}
            pg = f[key]
            for subkey in pg.keys():
                SNR_opt_info_loaded[int(key)][subkey] = {subsubkey: pg[subkey][subsubkey][()][0] for subsubkey in pg[subkey].keys()}
                
    return SNR_opt_info_loaded

#####################################################################################
# Catalog and events handling
#####################################################################################

def get_event(evs, idx):
    """
    Select events from a catalog by index.

    :param dict(array, array, ...) evs: The dictionary containing the parameters of the events, as in :py:data:`events`.
    :param list(int) or array(int) or int idx: The indexes of the events to select.
    
    :return: The dictionary containing the subsample of events.
    :rtype: dict(array, array, ...)

    """
    
    res = {k: np.squeeze(np.array([evs[k][idx], ] )) for k in evs.keys()}
    try:
        len(res['Mc'])
    except:
        res = {k: np.array( [res[k], ] )  for k in res.keys()}
    return res
        
def get_events_subset(evs, detected):
    """
    Select events from a catalog given condition.

    :param dict(array, array, ...) evs: The dictionary containing the parameters of the events, as in :py:data:`events`.
    :param list(bool) or array(bool) detected: Mask with the events to select, with the same shape as the arrays containing the events parameters.
    
    :return: The dictionary containing the subsample of events.
    :rtype: dict(array, array, ...)

    """
    return get_event(evs, np.argwhere(detected))    

def save_events(fname, data):
    """
    Store a dictionary containing the events parameters in ``h5`` file.
    
    :param str fname: The name of the file to store the events in. This has to include the path and the ``h5`` or ``hdf5`` extension.
    :param dict(array, array, ...) data: The dictionary containing the parameters of the events, as in :py:data:`events`.
    
    """
    print('Saving to %s '%fname)
    with h5py.File(fname, 'w') as out:
            
        def cd(n, d):
            d = np.array(d)
            out.create_dataset(n, data=d, compression='gzip', shuffle=True)
        
        for key in data.keys():
            cd(key, data[key])

#####################################################################################
# Actual computation of SNR timeseries
#####################################################################################

def main(idx, FLAGS):
        
        idx = idx-1        

        ti = time.time()
        Net= { k: copy.deepcopy(glob.detectors).pop(k) for k in FLAGS.net} 
        for i,psd in enumerate(FLAGS.psds): 
            Net[FLAGS.net[i]]['psd_path'] = psd#os.path.join(glob.detPath, psd)
        
        is_tidal, is_prec, is_HM, is_ecc = False, False, False, False
        if 'tidal' in FLAGS.lalargs:
            is_tidal = True
        if 'precessing' in FLAGS.lalargs:
            is_prec = True
        if 'HM' in FLAGS.lalargs:
            is_HM = True
        if 'eccentric' in FLAGS.lalargs:
            is_ecc = True
        wf_model = wf.LAL_WF(FLAGS.wf_model.split('-')[1], is_tidal=is_tidal, is_HigherModes=is_HM, is_Precessing=is_prec, is_eccentric=is_ecc)
        wf_model_name = FLAGS.wf_model
        
        if hasattr(wf_model, 'fRef'):
            wf_model.fRef = FLAGS.fmin
        dname = FLAGS.fout.split('/')[-1]
        if dname=='':
            dname = FLAGS.fout.split('/')[-2]
        
        print('\n------------ Results directory name:  ------------\n%s' %dname)
        print('--------------------------------------------------\n')
                
        logidx = '_'+str(FLAGS.idx_in)+'_to_'+str(FLAGS.idx_f) 
        if FLAGS.resume_run==0: 
            logfile = os.path.join(FLAGS.fout, 'logfile'+logidx+'_'+str(idx)+'.txt') #out_path+'logfile.txt'
        else:
            logfile = os.path.join(FLAGS.fout, 'logfile'+logidx+'_'+str(idx)+'_resume_'+str(FLAGS.resume_run)+'.txt')
        myLog = Logger(logfile)
        sys.stdout = myLog
        sys.stderr = myLog
        
        print('\n------------ Network used:  ------------\n%s' %str(Net))
        print('----------------------------------------\n')
    
        print('------ Waveform:------\n%s' %wf_model_name)
        print('----------------------\n')
        
        modes_used = FLAGS.modes_list
        if 'all' in modes_used:
            modes_used = None
        
        ts_simulator = tsCompute.simulate_SNR_timeseries(Net, 
                                                        is_ASD=FLAGS.is_ASD, 
                                                        fmin=FLAGS.fmin,
                                                        fmax=FLAGS.fmax, # in Hz
                                                        time_interval=FLAGS.time_interval, # in ms
                                                        df_integrals = FLAGS.df_integrals, # in Hz
                                                        reference_detector=FLAGS.reference_detector,
                                                        individual_modes=modes_used
                                                        )
        
        ti_evs=  time.time()
        for it in range( FLAGS.all_n_it_pools[idx] ):
            ti_loop=  time.time()
            idxin=FLAGS.idxs_lists[str(idx)][it][0] 
            idxf=FLAGS.idxs_lists[str(idx)][it][1]  
            
            ev_chunk = FLAGS.events_lists[str(idx)][it] 
            nevents_chunk = len(ev_chunk['dL'])
            
            i_in = idxin+FLAGS.idx_in
            i_f = idxf+FLAGS.idx_in
            suffstr = '_'+str(idxin+FLAGS.idx_in)+'_to_'+str(idxf+FLAGS.idx_in)

            if FLAGS.resume_run>0:
                if os.path.exists(os.path.join(FLAGS.fout, 'snrs_timeseries'+suffstr+'.hdf5')):
                    print('Batch already present, skipping...')
                    continue
            
            print('\nIn this chunk we have %s events, from %s to %s' %(nevents_chunk, i_in,  i_f  ))
            
            SNR_ts_chunk, SNR_sq_opt_chunk = {}, {}
            
            for iev in range(nevents_chunk):
                tmpev = {k: ev_chunk[k][iev] for k in ev_chunk.keys()}
                SNR_ts_chunk[iev+i_in], SNR_sq_opt_chunk[iev+i_in] = ts_simulator.injectSignal(tmpev, wf_model, df=1./8.)
            
            to_file_snr_timeseries(SNR_ts_chunk, FLAGS.fout, suff=suffstr, verbose=True)
            to_file_snr_opt_info(SNR_sq_opt_chunk, FLAGS.fout, suff=suffstr, verbose=True)

            print('------ Time to compute chunk %s: %s sec.' %(it, str(time.time()-ti_loop)))

        te=time.time()
        print('------')
        print('------ Time to compute events: %s sec.\n\n' %( str((te-ti_evs))))

        print('------ Done for CPU %s. ' %(idx))
        myLog.close()
        
def mainMPI(i):
    # Just a workaround to make mpi work without starmap
    return main(i, FLAGS)

parser = argparse.ArgumentParser(prog = 'compute_snrts_from_catalog.py', description='Executable to run ``SNR_timeseries`` on a catalog of events, with the possibility to parallelize over multiple CPUs, ready to use both on single machines and on clusters.')
parser.add_argument("--fname_obs", default='', type=str, required=True, help='Name of the file containing the catalog, without the extension ``h5``.')
parser.add_argument("--fname_snrs", default='', type=str, required=False, help='Name of the file containing the optimal SNRs associated with the catalog, with the extension ``txt``.')
parser.add_argument("--fout", default='test_ts', type=str, required=True, help='Path to output folder, which has to exist before the script is launched.')
parser.add_argument("--wf_model",  default='LAL-IMRPhenomXHM', type=str, required=False, help='Name of the waveform model.')
parser.add_argument("--batch_size", default=1, type=int, required=False, help='Size of the batch to be computed in vectorized form on each process.')
parser.add_argument("--npools", default=1, type=int, required=False, help='Number of parallel processes.')
parser.add_argument("--snr_th", default=12., type=float, required=False, help='Threshold value for the SNR to consider the event detectable. FIMs are computed only for events with SNR exceeding this value.')
parser.add_argument("--idx_in", default=0, type=int, required=False, help='Index of the event in the catalog from which to start the calculation.')
parser.add_argument("--idx_f", default=None, type=int, required=False, help='Index of the event in the catalog from which to end the calculation.')
parser.add_argument("--fmin", default=2., type=float, required=False, help='Minimum frequency of the grid, in Hz.')
parser.add_argument("--fmax", default=4096., type=float, required=False, help='Maximum frequency of the grid, in Hz.')
parser.add_argument("--net", nargs='+', default=['L1', ], type=str, required=False, help='The network of detectors to be used, separated by *single spacing*.')
parser.add_argument("--psds", nargs='+', default=['AplusDesign.txt', ], type=str, required=False, help='The paths to PSDs of each detector in the network inside the folder ``psds/``, separated by *single spacing*.')
parser.add_argument("--mpi", default=0, type=int, required=False, help='Int specifying if the code has to parallelize using multiprocessing (``0``), or using MPI (``1``), suitable for clusters.')
parser.add_argument("--lalargs", nargs='+', default=['HM'], type=str, required=False, help='Specifications of the waveform when using ``LAL`` interface, separated by *single spacing*.')
parser.add_argument("--resume_run", default=0, type=int, required=False, help='Int specifying whether to resume previous run. In this case the parent seed structure is preserved.')
parser.add_argument("--modes_list", nargs='+', default=['22', '33', '44'], type=str, required=False, help='The modes to be used for the waveform.')
parser.add_argument("--reference_detector", default='L1', type=str, required=False, help='The detector to be used as reference for the SNR time series.')
parser.add_argument("--is_ASD", default=True, type=bool, required=False, help='If ``True``, the PSDs are considered as ASD (Amplitude Spectral Density) instead of PSD (Power Spectral Density).')
parser.add_argument("--time_interval", default=40., type=float, required=False, help='Time interval in ms for the SNR time series.')
parser.add_argument("--df_integrals", default=1./4096., type=float, required=False, help='Frequency step for the integrals in Hz.')

if __name__ =='__main__':
    
    FLAGS = parser.parse_args()
    
    print('Input arguments: %s' %str(FLAGS))
    
    ti =  time.time()
    
    #####################################################################################
    # LOAD EVENTS
    #####################################################################################
    
    fname_obs = os.path.join(  FLAGS.fname_obs)
    #fname_obs = os.path.join('../data', FLAGS.fname_obs+'.h5')
    if not os.path.exists(fname_obs):
        raise ValueError('Path to catalog does not exist. Value entered: %s' %fname_obs)
    
    print('Loading events from %s...' %fname_obs)
    events_loaded = utils.load_population(fname_obs)
    
    keylist=list(events_loaded.keys())
    nevents_total = len(events_loaded[keylist[0]])
    print('This catalog has %s events.' %nevents_total)
    
    if FLAGS.fname_snrs != '':
        snrs_loaded = np.array([np.loadtxt(FLAGS.fname_snrs)])
    else:
        snrs_loaded = np.full_like(events_loaded[keylist[0]], FLAGS.snr_th + 1)
        
    if FLAGS.idx_f is None:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:] for k in events_loaded.keys()}
        snrs_loaded = snrs_loaded[FLAGS.idx_in:]
    else:
        events_loaded = {k: events_loaded[k][FLAGS.idx_in:FLAGS.idx_f] for k in events_loaded.keys()}
        snrs_loaded = snrs_loaded[FLAGS.idx_in:FLAGS.idx_f]
    nevents_total = len(events_loaded[keylist[0]])
    print('Using events between %s and %s, total %s events' %(FLAGS.idx_in, FLAGS.idx_f, nevents_total) )
    
    # Select the events with SNR above the threshold
    detected = snrs_loaded > FLAGS.snr_th
    events = get_events_subset(events_loaded, detected)
    print('Detected events with SNR > %s: %s' %(FLAGS.snr_th, np.sum(detected)))

    nevents_total = len(events[keylist[0]])
    
    if not 'ra' in events.keys():
        events['ra']  = events['phi']
        events['dec'] = 0.5*np.pi - events['theta']

        _ = events.pop('theta', None)
        _ = events.pop('phi', None)

    #########################################################################################
    # SPLIT EVENTS BETWEEN PROCESSES ACCORDING TO BATCH SIZE AND NUMBER OF PROCESSES REQUIRED
    #########################################################################################
    
    batch_size = FLAGS.batch_size
    npools = FLAGS.npools
    
    n_per_it = batch_size*FLAGS.npools # total events computed simultaneously on all cores
    if n_per_it>nevents_total:
        raise ValueError('The chosen batch size and number of pools are too large (it would take <1 iteration to cover all data). Choose smaller values.')
    
    n_it_per_pool = nevents_total//n_per_it # iterations done on every core
    nev_covered = n_per_it*n_it_per_pool # events done on every core
    #if nev_covered<nevents_total:
        
    nev_miss = nevents_total-nev_covered
    # compute how many pools will have to do one iteration more  
    diff = nevents_total-nev_covered
    
    if diff==0: # all cores do same number of iterations

        all_n_it_pools = [ n_it_per_pool for _ in range(npools) ]
        all_n_per_pool =  [ n_it_per_pool*batch_size for _ in range(npools)]
        full_last_chunk_sizes =  [ batch_size for _ in range(npools) ]
        #last_chunk_sizes = 
        cores_to_add=0
        n_it_per_pool_extra=0
        res=0
        int_part=0
        last_chunk_sizes=[]
    else:
        if diff<batch_size: 
            cores_to_add=1 # only 1 core needs to do one more iteration but with n of events < batch size
            res = 1
            int_part = 0
            n_it_per_pool_extra = 1
        elif diff==batch_size: # only one core needs to do one more iteration with n of events = batch size
            cores_to_add=1 
            res=0
            int_part = 1
            n_it_per_pool_extra = 1
        else:
            evsres = diff%batch_size # number of events left after cores did extra iteration 
            int_part = int((diff-evsres)/batch_size) # number of cores doing exactly one batch more
            if evsres>0:
                res=1
            else: res=0
                
            cores_to_add =  int(int_part+res)
            #if 
            n_it_per_pool_extra = 1
        print('Cores which do one extra iteration: %s' %cores_to_add)
        print('N of cores doing less than one batch more: %s' %res)
        print('N of cores doing exactly one batch more: %s' %int_part)
            
        last_chunk_sizes = [ batch_size for _ in range(int_part) ]
        if int(nev_miss-batch_size*int_part)>0:
            last_chunk_sizes+=[ int(nev_miss-batch_size*int_part) for _ in range(1)] 
        print('Last chunk size will be %s on last %s cores' %(str(last_chunk_sizes), cores_to_add))
        all_n_it_pools = [ n_it_per_pool for _ in range(npools-cores_to_add) ]+[ n_it_per_pool+n_it_per_pool_extra for _ in range(cores_to_add)]
        all_n_per_pool = [ n_it_per_pool*batch_size for _ in range(npools-cores_to_add)]+[ (n_it_per_pool*batch_size)+last_chunk_sizes[i] for i in range(cores_to_add)] # events computed on every core in total 
        full_last_chunk_sizes = np.hstack( [np.zeros(npools-len(last_chunk_sizes)), last_chunk_sizes] ) #.T
    
    print('Number of iterations on every core: %s' %str(all_n_it_pools))     
    print('%s iterations will be used to cover all the %s events.' %(max(all_n_it_pools), nevents_total))
    print('For each iteration, we will parallise on %s cores with max %s events/core.' %(npools, batch_size))
    print('Number of events computed on every core in total: %s' %str(all_n_per_pool))

    all_n_it_pools = np.array(all_n_it_pools)

    all_batch_sizes = np.zeros((npools, max(all_n_it_pools) )).T
    for i in range(npools):
        for j in range(max(all_n_it_pools)):
            #print(i, j)
            is_last = j==max(np.array(all_n_it_pools)-1)
            #print(is_last)
            if not is_last or np.all(full_last_chunk_sizes==0):
                all_batch_sizes[j, i] = batch_size
            else:
                all_batch_sizes[j, i] = full_last_chunk_sizes[i]
    
    all_batch_sizes = all_batch_sizes.astype('int')
    print('All batch sizes, last three iterations (shape nbatches x npools ): ')
    print(all_batch_sizes[-3:, :])
    
    assert all_batch_sizes.sum()==nevents_total
    assert np.array(all_n_per_pool).sum()==nevents_total
    
    events_lists = {str(i):[] for i in range(npools)}
    idxs_lists = {str(i):[] for i in range(npools)}
    
    pin=0
    for it in range(all_batch_sizes.shape[0]): # iterations
        for p in range(all_batch_sizes.shape[-1]): # pools
            pf =  pin+all_batch_sizes[it, p]
            if pf>pin:
                events_lists[str(p)].append({k: events[k][pin:pf] for k in events.keys()})
                idxs_lists[str(p)].append( (pin, pf) )
                nevents_chunk = len(events_lists[str(p)][-1]['dL'])
                assert nevents_chunk == all_batch_sizes[it, p]
                pin = pf
    ncheck=0       
    for evl in  events_lists.values():
        for evs in evl:
            ncheck+=len(evs['dL'])
        
    assert ncheck==nevents_total
    
    FLAGS.all_n_it_pools = all_n_it_pools
    FLAGS.events_lists = events_lists
    FLAGS.idxs_lists = idxs_lists
    
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
        (main(1, FLAGS), )
        
    ############################################################################
    # Concatenate results and clean
    ############################################################################
    
    print('\nSaving final version to file...')
    if FLAGS.idx_f is None:
            idxf = str(nevents_total)
    else: idxf = FLAGS.idx_f
        
    suffstr = '_'+str(FLAGS.idx_in)+'_to_'+str(idxf)      

    pin=FLAGS.idx_in
    SNR_ts, SNR_sq_opt = {}, {}
    for it in range(all_batch_sizes.shape[0]): # iterations
        for p in range(all_batch_sizes.shape[-1]): # pools
            pf =  pin+all_batch_sizes[it, p]
            if pf>pin:
                print('Concatenating files from %s to %s' %(pin, pf))
                suff_batch = '_'+str(pin)+'_to_'+str(pf)
                tmp_SNR_ts = from_file_snr_timeseries(FLAGS.fout, suff=suff_batch)
                tmp_SNR_sq_opt = from_file_snr_opt_info(FLAGS.fout, suff=suff_batch)
                for k in tmp_SNR_ts.keys():
                    SNR_ts[k] = tmp_SNR_ts[k]
                
                for k in tmp_SNR_sq_opt.keys():
                    SNR_sq_opt[k] = tmp_SNR_sq_opt[k]

                pin = pf
    
    to_file_snr_timeseries(SNR_ts, FLAGS.fout, suff=suffstr, verbose=True)
    to_file_snr_opt_info(SNR_sq_opt, FLAGS.fout, suff=suffstr, verbose=True)
    
    print('Saving catalog of detected events...')
    events['snr'] = snrs_loaded[detected]
    fname_out = os.path.join(FLAGS.fout, 'detected_events'+suffstr+'.hdf5')
    save_events(fname_out, events)
    idxs_det = np.arange(FLAGS.idx_in, int(idxf))[np.argwhere(snrs_loaded>FLAGS.snr_th)]
    np.savetxt(os.path.join(FLAGS.fout, 'idxs_det'+suffstr+'.txt'), idxs_det) 
    
    if (FLAGS.npools>1) or (FLAGS.npools==1 and all_n_it_pools[0]>1): 
        print('Cleaning...')
        pin=FLAGS.idx_in
        for it in range(all_batch_sizes.shape[0]): # iterations
                for p in range(all_batch_sizes.shape[-1]): # pools
                    pf =  pin+all_batch_sizes[it, p]
                    if pf>pin:
                        suffstr = '_'+str(pin)+'_to_'+str(pf)
                        try:
                            os.remove(os.path.join(FLAGS.fout, 'snrs_timeseries'+suffstr+'.hdf5'))
                            os.remove(os.path.join(FLAGS.fout, 'snrs_opt_info'+suffstr+'.hdf5'))
                        except:
                            print('Files %s and %s not found, skipping...' %(os.path.join(FLAGS.fout, 'snrs_timeseries'+suffstr+'.hdf5'), os.path.join(FLAGS.fout, 'snrs_opt_info'+suffstr+'.hdf5')))
                        pin = pf
                        
    print('Done! Total time: %s sec.' %(str(time.time()-ti)))