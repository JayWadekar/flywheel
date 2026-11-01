"""Matched-filter SNR of every event of the noisy ('ML') data set: the network peak of the 22-mode SNR time series,
rho_mf = sqrt(sum_det max_t |rho_22,det(t)|^2). The campaign selects events on rho_mf (data only), not on the
optimal SNR (a function of the truth).

Usage: python compute_rho_mf.py  -> RESULTS/rho_mf_ML.npy (one entry per row of the ML data set)
"""
import os
import numpy as np, h5py
import config
import o5_data as od

if __name__ == '__main__':
    data = od.O5Data('ML')
    rho = np.zeros(data.n_events)
    for iev in range(data.n_events):
        g = data.ts[data.key(iev)]
        rho[iev] = np.sqrt(sum(np.max(np.abs(g[d]['22'][()]))**2 for d in g if isinstance(g[d], h5py.Group)))
    os.makedirs(config.RESULTS, exist_ok=True)
    np.save(os.path.join(config.RESULTS, 'rho_mf_ML.npy'), rho)
    print(f'{data.n_events} events, {np.sum(rho > 11)} with rho_mf > 11')
