"""Self-models trained on healthy data only (importable; no argparse).

one-step:   (s_t, a_t)                                   -> s_{t+1} - s_t
multi-step: (s_{t-9..t}, a_{t-9..t-1}, a_{t..t+9})       -> s_{t+1..t+10} - s_t
MLP 3 x 512, GELU, AdamW, inputs and outputs normalized. Errors are squared,
per state dimension, divided by the healthy per-dimension state std.
"""
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn

import common as C


class MLP(nn.Module):
    out: int

    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = nn.gelu(nn.Dense(512)(x))
        return nn.Dense(self.out)(x)


def windows(obs, act, kind, ts):
    """X, Y for every trial and every prediction time t in ts. Shapes (n, len(ts), dim)."""
    ts = np.asarray(ts)
    if kind == "one":
        X = np.concatenate([obs[:, ts], act[:, ts]], -1)
        Y = obs[:, ts + 1] - obs[:, ts]
        return X, Y
    n = obs.shape[0]
    S = np.stack([obs[:, ts - k] for k in range(C.H - 1, -1, -1)], 2)              # s_{t-9..t}
    Ap = np.stack([act[:, ts - k] for k in range(C.H - 1, 0, -1)], 2)              # a_{t-9..t-1}
    Af = np.stack([act[:, ts + k] for k in range(C.H)], 2)                         # a_{t..t+9}
    X = np.concatenate([S.reshape(n, len(ts), -1), Ap.reshape(n, len(ts), -1), Af.reshape(n, len(ts), -1)], -1)
    Y = (np.stack([obs[:, ts + k] for k in range(1, C.H + 1)], 2) - obs[:, ts][:, :, None]).reshape(n, len(ts), -1)
    return X, Y


def valid_ts(kind, T=C.T_TOTAL):
    return np.arange(0, T) if kind == "one" else np.arange(C.H - 1, T - C.H + 1)


def train(kind, seed, obs, act):
    X, Y = windows(obs, act, kind, valid_ts(kind))
    X, Y = X.reshape(-1, X.shape[-1]).astype(np.float32), Y.reshape(-1, Y.shape[-1]).astype(np.float32)
    norm = dict(xm=X.mean(0), xs=X.std(0) + 1e-6, ym=Y.mean(0), ys=Y.std(0) + 1e-6,
                state_std=obs.reshape(-1, obs.shape[-1]).std(0) + 1e-6)
    Xn, Yn = (X - norm["xm"]) / norm["xs"], (Y - norm["ym"]) / norm["ys"]
    model = MLP(Y.shape[1])
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.zeros((1, X.shape[1])))
    bs = 1024
    steps = C.SM_EPOCHS * max(1, len(Xn) // bs)
    opt = optax.adamw(optax.cosine_decay_schedule(1e-3, steps), weight_decay=1e-4)
    st = opt.init(params)

    @jax.jit
    def upd(params, st, x, y):
        loss, g = jax.value_and_grad(lambda p: ((model.apply(p, x) - y) ** 2).mean())(params)
        u, st = opt.update(g, st, params)
        return optax.apply_updates(params, u), st, loss

    rng = np.random.default_rng(seed)
    for ep in range(C.SM_EPOCHS):
        perm = rng.permutation(len(Xn)); tot = 0.0
        for i in range(0, len(perm) - bs + 1, bs):
            b = perm[i:i + bs]
            params, st, l = upd(params, st, jnp.asarray(Xn[b]), jnp.asarray(Yn[b])); tot += float(l)
        print(f"    {kind} seed {seed} epoch {ep + 1}/{C.SM_EPOCHS} loss {tot / max(1, len(perm) // bs):.4f}", flush=True)
    return dict(kind=kind, seed=seed, params=params, norm=norm, out=Y.shape[1])


def get(kind, seed, obs, act):
    path = os.path.join(C.OUT, f"selfmodel_{kind}_s{seed}.pkl")
    if os.path.exists(path):
        return pickle.load(open(path, "rb"))
    m = train(kind, seed, obs, act)
    pickle.dump(m, open(path, "wb"))
    return m


def errors(m, obs, act, ts):
    """Squared normalized error per trial, per t, per state dim: (n, len(ts), 27)."""
    X, Y = windows(obs, act, m["kind"], ts)
    n, nt = X.shape[:2]
    nm = m["norm"]
    model = MLP(m["out"])
    f = jax.jit(lambda x: model.apply(m["params"], x))
    Xf = ((X.reshape(-1, X.shape[-1]) - nm["xm"]) / nm["xs"]).astype(np.float32)
    P = np.concatenate([np.asarray(f(jnp.asarray(Xf[i:i + 65536]))) for i in range(0, len(Xf), 65536)])
    P = P * nm["ys"] + nm["ym"]
    E = ((P - Y.reshape(len(P), -1)) ** 2).reshape(n, nt, -1)
    if m["kind"] == "multi":
        E = E.reshape(n, nt, C.H, -1).mean(2)                     # average over the 10 horizons
    return E / nm["state_std"] ** 2
