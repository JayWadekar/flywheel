"""Vectorized replacement for cogwheel's per-sample distance draw.

cogwheel's ``BaseCoherentScore._sample_distance`` is ``np.vectorize(LookupTable.sample_distance)``: for every
posterior sample it builds a distance grid, a k=1 spline and its antiderivative (~0.4 ms per sample, ~40% of the
per-event cost for 4 x 10^4 draws). ``sample_distance`` below does the same computation for all samples at once:
the same broad + focused grids (``resolution`` points each), the same integrand ``_function_integrand``, the
trapezoidal cumulative (identical to the antiderivative of a k=1 spline at the nodes) and linear inverse-CDF
interpolation of a uniform draw. Only the random stream differs.

Usage: ``fast_distance.patch(cs_instance)`` replaces ``cs_instance._sample_distance``.
"""
import numpy as np


def sample_distance(table, d_h, h_h, rng, resolution=256, chunk=4096):
    d_h, h_h = np.broadcast_arrays(np.asarray(d_h, float), np.asarray(h_h, float))
    shape = d_h.shape
    d_h, h_h = d_h.ravel(), h_h.ravel()
    out = np.empty(d_h.size)
    dmax = table.d_luminosity_max
    broad = np.linspace(0, dmax, resolution)[1:]
    lin = np.linspace(0, 1, resolution)
    for s in range(0, d_h.size, chunk):
        dh, hh = d_h[s:s + chunk], h_h[s:s + chunk]
        n = len(dh)
        # focused grid: uniform in u = 1/d over u_peak +- 10 sigma (cogwheel: sigmas=10)
        bounds = table._get_distance_bounds(dh, hh, sigmas=10.)          # (2, n)
        u0, u1 = 1/bounds[0], 1/bounds[1]
        focused = 1/(u0[:, None] + (u1 - u0)[:, None]*lin[None, :])
        # cogwheel drops focused points outside (0, dmax); move them onto dmax (a node of the broad grid), which
        # adds only zero-width intervals
        focused = np.where((focused > 0) & (focused < dmax), focused, dmax)
        d = np.sort(np.concatenate([np.broadcast_to(broad, (n, len(broad))), focused], axis=1), axis=1)
        post = table._function_integrand(d, dh[:, None], hh[:, None])
        cum = np.concatenate([np.zeros((n, 1)), np.cumsum(0.5*(post[:, 1:] + post[:, :-1])*np.diff(d, axis=1),
                                                          axis=1)], axis=1)
        target = rng.uniform(0, 1, n)*cum[:, -1]
        j = np.clip((cum < target[:, None]).sum(axis=1), 1, d.shape[1] - 1)   # cum[j-1] < target <= cum[j]
        r = np.arange(n)
        c0, c1 = cum[r, j - 1], cum[r, j]
        frac = np.where(c1 > c0, (target - c0)/np.where(c1 > c0, c1 - c0, 1), 0.)
        out[s:s + chunk] = d[r, j - 1] + frac*(d[r, j] - d[r, j - 1])
    return out.reshape(shape) if shape else out[()]


def patch(cs_instance):
    table, rng = cs_instance.lookup_table, cs_instance.lookup_table._rng
    cs_instance._sample_distance = lambda d_h, h_h: sample_distance(table, d_h, h_h, rng)
    return cs_instance
