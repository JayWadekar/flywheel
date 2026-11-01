"""Run the O5 P-P campaign on a random subset of events with noisy 22-mode network peak SNR > RHO_CUT.

The catalog's detection cut is on the *optimal* SNR (> 8), a function of the true parameters. Cutting on the
noisy matched-filter SNR rho_mf > 11 makes the selection effectively data-only: an event with rho_opt < 8 needs
a >~2.4 sigma noise fluctuation to pass (rho_mf - rho_opt = +0.8 +- 0.9 near threshold).

Events where the ML search did not converge (ML_fallback = 1: the template is the injection) are excluded.
The event list is a fixed permutation (SEED), so a larger N extends a smaller run with the same first events;
it is written to OUT/events_N{N}_seed{SEED}.txt (the flow is trained with these events held out).

Set the thread variables (OMP_NUM_THREADS=1 etc.) before running, or each worker starts one BLAS thread per core.

Usage: PP_CLOUD=rb|flow [PP_OUT=dir] python run_pp_campaign.py N_EVENTS [N_PROCS] [SEED]
(events already done in OUT are skipped; needs rho_mf_ML.npy from compute_rho_mf.py)
"""
import sys, os, time
import numpy as np
from multiprocessing import Pool
import config

RHO_CUT = 11.
OUT = os.environ.get('PP_OUT', os.path.join(config.RESULTS, f"pp_O5_ML_{os.environ.get('PP_CLOUD', 'rb')}"))


def work(iev):
    import run_pp_event
    try:
        return run_pp_event.process(iev, OUT)
    except Exception as exc:  # keep the campaign going; record the failure
        with open(os.path.join(OUT, f'failed_{iev:05d}.txt'), 'w') as f:
            f.write(repr(exc))
        return iev, -1


if __name__ == '__main__':
    n_events = int(sys.argv[1])
    n_procs = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
    import o5_data as od
    rho = np.load(os.path.join(config.RESULTS, 'rho_mf_ML.npy'))
    pool_idx = np.where(rho > RHO_CUT)[0]
    pool_idx = pool_idx[od.O5Data('ML').ml['ML_fallback'][pool_idx] == 0]
    # A fixed permutation, so a larger N extends a smaller run with the same first events
    order = np.random.default_rng(seed).permutation(pool_idx)
    chosen = np.sort(order[:n_events])
    os.makedirs(OUT, exist_ok=True)
    np.savetxt(os.path.join(OUT, f'events_N{n_events}_seed{seed}.txt'), chosen, fmt='%d')
    todo = [i for i in chosen if not os.path.exists(os.path.join(OUT, f'event_{i:05d}.h5'))]
    print(f'{len(pool_idx)} events with rho_mf > {RHO_CUT}; running {len(todo)} of {n_events} on {n_procs} processes', flush=True)
    t0 = time.time()
    with Pool(n_procs, maxtasksperchild=20) as pool:
        for k, (iev, dt) in enumerate(pool.imap_unordered(work, todo), 1):
            if k % 10 == 0 or dt < 0:
                print(f'[{time.time() - t0:6.0f} s] {k}/{len(todo)} done (event {iev}: {dt:.0f} s)', flush=True)
    print('finished in %.0f s' % (time.time() - t0))
