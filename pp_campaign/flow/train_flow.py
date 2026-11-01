"""Conditional normalizing flow for the low-latency intrinsic cloud, trained on make_training_set.py output.

Context (what the search provides): ML template ln Mc, eta, chi1z, chi2z, ln dL_ML, ln rho_mf, ln R33_ML, ln R44_ML.
Target (7 numbers per draw): offsets (true - ML) of ln Mc, logit(eta/0.25), chi1z, chi2z; offsets of ln R33, ln R44;
and the template match rho(true)/rho(ML). A draw is one cloud member: intrinsic parameters, the mode ratios R33,
R44 of that template (no waveform calls), and the factor by which its data overlap differs from the ML template's
(the distance correction, no data evaluation).

Over events the true parameters are draws from p(theta | search output), so the trained density is a calibrated
posterior by construction (neural posterior estimation). Checked here on the held-out P-P events: probability
integral transform (PIT) of the truth in every target marginal, and in m2_det.

Caveat: the simulated search (SNRtsML.py) starts its optimizer at the true parameters and stops part-way along the
likelihood valley, so its ML template sits ~0.5 posterior sigma from the truth instead of ~1 sigma. The flow learns
this and its m2 draws are narrower than a real search would allow. A flow for a real search must be trained on that
search's own outputs (e.g. its template bank).

Defaults are the configuration used for the paper (best validation loss in a sweep): 4 autoregressive spline
transforms, 64 hidden units, AdamW with learning rate 5e-4 and weight decay 1e-3, 300 epochs.

Usage: python train_flow.py HOLDOUT_EVENTS.txt [n_epochs]
       HOLDOUT_EVENTS.txt: the P-P event list written by run_pp_campaign.py (excluded from training)
       -> RESULTS/flow/flow.pt, flow_check.pdf, flow_check.txt
Environment: FLOW_TRANSFORMS, FLOW_HIDDEN, FLOW_LR, FLOW_WD (architecture/optimizer), FLOW_TAG (file suffix)
"""
import os, sys, time
import numpy as np
import torch, zuko
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RES = os.path.join(config.RESULTS, 'flow')
NAMES = ['dlnMc', 'dlogit_eta', 'dchi1z', 'dchi2z', 'dlnR33', 'dlnR44', 'match']
DIM = 7
TRANSFORMS = int(os.environ.get('FLOW_TRANSFORMS', 4))
HIDDEN = int(os.environ.get('FLOW_HIDDEN', 64))
LR, WD = float(os.environ.get('FLOW_LR', 5e-4)), float(os.environ.get('FLOW_WD', 1e-3))
FLOW_FILE = 'flow' + os.environ.get('FLOW_TAG', '') + '.pt'
torch.set_num_threads(8)


def logit_eta(eta):
    e = np.clip(eta/0.25, 1e-4, 1 - 1e-6)
    return np.log(e/(1 - e))


def inv_logit_eta(u):
    return 0.25/(1 + np.exp(-u))


def context(t):
    """(n, 8) context from the search output."""
    return np.c_[np.log(t['ml_Mc']), logit_eta(t['ml_eta']), t['ml_chi1z'], t['ml_chi2z'], np.log(t['ml_dL']),
                 np.log(t['rho_mf']), np.log(t['R33_ml']), np.log(t['R44_ml'])]


def target(t):
    """(n, 7) targets: offsets of the true template from the ML template, and the match."""
    return np.c_[np.log(t['true_Mc']) - np.log(t['ml_Mc']), logit_eta(t['true_eta']) - logit_eta(t['ml_eta']),
                 t['true_chi1z'] - t['ml_chi1z'], t['true_chi2z'] - t['ml_chi2z'],
                 np.log(t['R33']/t['R33_ml']), np.log(t['R44']/t['R44_ml']), t['rho_true_profile']/t['rho_ml_profile']]


class IntrinsicFlow:
    """Trained flow plus normalization; `sample(ctx, n)` returns the cloud for one event."""

    def __init__(self, cx_mean, cx_std, y_mean, y_std, state=None, transforms=4, hidden=128):
        self.cx_mean, self.cx_std, self.y_mean, self.y_std = (np.asarray(a, np.float32) for a in (cx_mean, cx_std, y_mean, y_std))
        self.dim = len(self.y_mean)
        self.arch = {'transforms': transforms, 'hidden': hidden}
        self.flow = zuko.flows.NSF(self.dim, 8, transforms=transforms, hidden_features=(hidden, hidden))
        if state is not None:
            self.flow.load_state_dict(state)

    def sample(self, ctx, n=1000, seed=None, inflate=None):
        """ctx: (8,) context of one event -> dict with Mc, eta, chi1z, chi2z, R33, R44, match (arrays of n).
        inflate: optional per-target factors that widen the draws about their mean (calibrate_width.py)."""
        ctx = np.asarray(ctx, float)
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        if g is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            c = torch.tensor((ctx - self.cx_mean)/self.cx_std, dtype=torch.float32)
            y = self.flow(c).sample((n,)).numpy()*self.y_std + self.y_mean
        if inflate is not None:
            m = y.mean(0)
            y = m + (y - m)*np.asarray(inflate)[:y.shape[1]]
        out = {'Mc': np.exp(ctx[0] + y[:, 0]), 'eta': inv_logit_eta(ctx[1] + y[:, 1]),
               'chi1z': ctx[2] + y[:, 2], 'chi2z': ctx[3] + y[:, 3]}
        if self.dim == 7:
            out.update({'R33': np.exp(ctx[6] + y[:, 4]), 'R44': np.exp(ctx[7] + y[:, 5]), 'match': y[:, 6]})
        return out

    def save(self, fn):
        torch.save({'state': self.flow.state_dict(), 'cx_mean': self.cx_mean, 'cx_std': self.cx_std,
                    'y_mean': self.y_mean, 'y_std': self.y_std, **self.arch}, fn)

    @classmethod
    def load(cls, fn):
        d = torch.load(fn, weights_only=False)
        return cls(d['cx_mean'], d['cx_std'], d['y_mean'], d['y_std'], d['state'],
                   d.get('transforms', 4), d.get('hidden', 128))


def m2_det(Mc, eta):
    sq = np.sqrt(np.clip(1 - 4*eta, 0, None))
    q = (1 - sq)/(1 + sq)            # eta = q/(1+q)^2, q <= 1
    return q/(1 + q)*Mc/eta**0.6


def main(holdout, n_epochs=300):
    t = dict(np.load(os.path.join(RES, 'training_set.npz')))
    X, Y = context(t), target(t)[:, :DIM]
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1) & (t['ml_ML_fallback'] == 0) & (t['rho_mf'] > 8)
    t['is_test'] = np.isin(np.arange(len(ok)), np.loadtxt(holdout).astype(int))
    train = ok & ~t['is_test']
    test = ok & t['is_test']
    print(f'{ok.sum()} usable rows, {train.sum()} for training, {test.sum()} held-out P-P events', flush=True)
    cm, cs = X[train].mean(0), X[train].std(0)
    ym, ys = Y[train].mean(0), Y[train].std(0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(np.where(train)[0])
    nval = len(idx)//10
    val, tr = idx[:nval], idx[nval:]
    f = lambda a, m, s: torch.tensor((a - m)/s, dtype=torch.float32)
    Xt, Yt, Xv, Yv = f(X[tr], cm, cs), f(Y[tr], ym, ys), f(X[val], cm, cs), f(Y[val], ym, ys)
    model = IntrinsicFlow(cm, cs, ym, ys, transforms=TRANSFORMS, hidden=HIDDEN)
    opt = torch.optim.AdamW(model.flow.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    best, best_state, t0 = np.inf, None, time.time()
    for ep in range(n_epochs):
        model.flow.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = -model.flow(Xt[b]).log_prob(Yt[b]).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.flow.parameters(), 5.); opt.step()
        sched.step()
        model.flow.eval()
        with torch.no_grad():
            lv = -model.flow(Xv).log_prob(Yv).mean().item()
        if lv < best:
            best, best_state = lv, {k: v.clone() for k, v in model.flow.state_dict().items()}
        if ep % 5 == 0 or ep == n_epochs - 1:
            print(f'epoch {ep:3d}  train {loss.item():.3f}  val {lv:.3f}  best {best:.3f}  ({time.time() - t0:.0f} s)', flush=True)
    model.flow.load_state_dict(best_state)
    print(f'best validation loss {best:.4f}', flush=True)
    model.save(os.path.join(RES, FLOW_FILE))

    # Calibration on the held-out P-P events: PIT of the truth in each marginal
    ti = np.where(test)[0]
    pit = {k: [] for k in NAMES[:DIM] + ['m2_det']}
    widths = []
    t1 = time.time()
    for j in ti:
        s = model.sample(X[j], 1000, seed=int(j))
        yt = Y[j]
        truth = dict(zip(NAMES, yt))
        samp = {'dlnMc': np.log(s['Mc']) - X[j, 0], 'dlogit_eta': logit_eta(s['eta']) - X[j, 1],
                'dchi1z': s['chi1z'] - X[j, 2], 'dchi2z': s['chi2z'] - X[j, 3]}
        if DIM == 7:
            samp.update({'dlnR33': np.log(s['R33']) - X[j, 6], 'dlnR44': np.log(s['R44']) - X[j, 7], 'match': s['match']})
        for k in NAMES[:DIM]:
            pit[k].append(np.mean(samp[k] < truth[k]))
        pit['m2_det'].append(np.mean(m2_det(s['Mc'], s['eta']) < t['true_m2_src'][j]*(1 + t['true_z'][j])))
        widths.append(np.std(m2_det(s['Mc'], s['eta'])))
    print(f'sampling: {(time.time() - t1)/len(ti)*1e3:.1f} ms per event (1000 draws)')
    lines = []
    for k, v in pit.items():
        v = np.array(v)
        z = stats.norm.ppf(np.clip(v, 1e-4, 1 - 1e-4))
        lines.append(f'{k:11s} N={len(v)}  KS p {stats.kstest(v, "uniform").pvalue:.3f}  offset {z.mean():+.3f} +- {1/np.sqrt(len(v)):.3f}  sd {z.std():.3f}')
    print('\n'.join(lines))
    open(os.path.join(RES, FLOW_FILE.replace('.pt', '_check.txt')), 'w').write('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for k, v in pit.items():
        v = np.sort(v)
        ax.plot(np.r_[0, v, 1], np.r_[0, np.arange(1, len(v) + 1)/len(v), 1], lw=1.2, label=k)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('C.I.'); ax.set_ylabel('Fraction of events in C.I.'); ax.legend(fontsize=8)
    ax.set_title(f'Flow calibration, {len(ti)} held-out events')
    fig.savefig(os.path.join(RES, FLOW_FILE.replace('.pt', '_check.pdf')), bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300)
