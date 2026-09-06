"""Functional ensemble forward models for Rung 3.

One implementation covers three needs:

  * a single model            (E = 1)
  * a bootstrap ensemble      (E = 5, different inits + resampled data)
  * MAML / GrBAL meta-learning (parameters are plain tensors, so the inner
    adaptation step is differentiable and the outer loop can backprop through it)

Parameters are explicit tensors with a leading ensemble dimension, and the
forward pass is a batched matmul, so all E members are evaluated in ONE call
rather than a Python loop over E models. That keeps ensemble planning roughly
1.5-2x the cost of single-model planning instead of 5x.

Architecture matches Rung 1 exactly (4 -> 64 -> 64 -> 3, ReLU, delta targets)
so results remain comparable across rungs.
"""
import numpy as np
import torch

DIN, HID, DOUT = 4, 64, 3
TARGET = torch.tensor([1.0, 0.0, 0.0])


# ---------------------------------------------------------------- parameters
def init_params(E, seed=0, device="cpu"):
    """E independent parameter sets, each initialised like nn.Linear defaults."""
    g = torch.Generator(device=device).manual_seed(seed)
    p = []
    for fan_in, fan_out in ((DIN, HID), (HID, HID), (HID, DOUT)):
        bound = 1.0 / np.sqrt(fan_in)
        W = (torch.rand(E, fan_in, fan_out, generator=g, device=device) * 2 - 1) * bound
        b = (torch.rand(E, 1, fan_out, generator=g, device=device) * 2 - 1) * bound
        p += [W.requires_grad_(True), b.requires_grad_(True)]
    return p


def forward(x, params):
    """x: (E, B, DIN) -> (E, B, DOUT). Predicts the DELTA, not the next state."""
    W1, b1, W2, b2, W3, b3 = params
    h = torch.relu(torch.baddbmm(b1, x, W1))
    h = torch.relu(torch.baddbmm(b2, h, W2))
    return torch.baddbmm(b3, h, W3)


def member_loss(params, X, Y):
    """Sum over members of each member's own mean squared error.

    Summing (rather than averaging) over the ensemble dimension keeps every
    member's gradient the same size it would be if trained alone.
    """
    return ((forward(X, params) - Y) ** 2).mean(dim=(1, 2)).sum()


def clone(params):
    return [p.detach().clone().requires_grad_(True) for p in params]


# ---------------------------------------------------------------- plain training
def to_tensors(S, A, S2, E=1, idx=None):
    """Build (E, n, DIN) inputs and (E, n, DOUT) delta targets.

    idx: optional (E, n) index array, used for per-member bootstrap resampling.
    """
    X = np.concatenate([S, A], axis=1).astype(np.float32)
    Y = (S2 - S).astype(np.float32)
    if idx is None:
        Xe = np.repeat(X[None], E, axis=0)
        Ye = np.repeat(Y[None], E, axis=0)
    else:
        Xe = X[idx]
        Ye = Y[idx]
    return torch.from_numpy(Xe), torch.from_numpy(Ye)


def train_ensemble(S, A, S2, E=5, epochs=1000, lr=1e-3, seed=0, bootstrap=True,
                   params=None):
    """Train E members. Each sees its own bootstrap resample when bootstrap=True.

    Bootstrap resampling is what makes members disagree for reasons other than
    initialisation, which is what the disagreement signal needs to be meaningful.
    Pass `params` to fine-tune from existing weights instead of from scratch.
    """
    n = len(S)
    idx = None
    if bootstrap and E > 1:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(E, n))
    X, Y = to_tensors(S, A, S2, E=E, idx=idx)
    p = init_params(E, seed=seed) if params is None else clone(params)
    opt = torch.optim.Adam(p, lr=lr)
    for _ in range(epochs):
        loss = member_loss(p, X, Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return [t.detach() for t in p]


# ---------------------------------------------------------------- meta-training
def inner_adapt(params, X, Y, alpha, steps=1, create_graph=False):
    """GrBAL/MAML inner loop: gradient steps from `params` on a small window."""
    fast = params
    for _ in range(steps):
        loss = member_loss(fast, X, Y)
        grads = torch.autograd.grad(loss, fast, create_graph=create_graph)
        fast = [p - alpha * g for p, g in zip(fast, grads)]
    return fast


def meta_train(tasks, E=5, K=10, iters=3000, alpha=0.01, lr=1e-3, inner_steps=1,
               seed=0, second_order=True, log_every=None):
    """MAML-style meta-training over a distribution of bodies.

    tasks: list of (S, A, S2) arrays, one contiguous trajectory per body.

    Each iteration, every ensemble member independently samples a task and a
    contiguous window: the first K transitions are the adaptation set, the next
    K are the evaluation set. The outer loss is the post-adaptation error, so
    the learned initialisation is one from which K samples are *maximally
    informative* — not one that is accurate on average.
    """
    rng = np.random.default_rng(seed)
    p = init_params(E, seed=seed)
    opt = torch.optim.Adam(p, lr=lr)
    T = len(tasks)
    hist = []

    for it in range(iters):
        Xa, Ya, Xe, Ye = [], [], [], []
        for _ in range(E):
            S, A, S2 = tasks[rng.integers(T)]
            i = rng.integers(0, len(S) - 2 * K)
            xa, ya = to_tensors(S[i:i + K], A[i:i + K], S2[i:i + K], E=1)
            xe, ye = to_tensors(S[i + K:i + 2 * K], A[i + K:i + 2 * K],
                                S2[i + K:i + 2 * K], E=1)
            Xa.append(xa); Ya.append(ya); Xe.append(xe); Ye.append(ye)
        Xa, Ya = torch.cat(Xa), torch.cat(Ya)
        Xe, Ye = torch.cat(Xe), torch.cat(Ye)

        fast = inner_adapt(p, Xa, Ya, alpha, inner_steps, create_graph=second_order)
        outer = member_loss(fast, Xe, Ye)
        opt.zero_grad(); outer.backward(); opt.step()

        if log_every and (it + 1) % log_every == 0:
            hist.append((it + 1, float(outer) / E))
            print(f"   meta-iter {it + 1:>5}/{iters}  post-adapt loss "
                  f"{float(outer) / E:.6f}", flush=True)
    return [t.detach() for t in p], hist


def online_adapt(params, window, alpha, steps=1, E=None):
    """Adapt on a sliding window of recent transitions, at control time.

    Always starts from the meta-parameters rather than compounding, which is
    what GrBAL does: the initialisation is the thing that was optimised for
    K-sample adaptation, so re-adapting from it each step is the intended use.
    """
    S, A, S2 = window
    E = params[0].shape[0] if E is None else E
    X, Y = to_tensors(S, A, S2, E=E)
    with torch.enable_grad():
        p = [t.detach().requires_grad_(True) for t in params]
        fast = inner_adapt(p, X, Y, alpha, steps, create_graph=False)
    return [t.detach() for t in fast]


# ---------------------------------------------------------------- evaluation
def predict_next(params, state, action):
    """One-step next-state prediction, ensemble mean. For diagnostics."""
    x = np.concatenate([np.asarray(state), np.asarray(action)]).astype(np.float32)
    E = params[0].shape[0]
    xt = torch.from_numpy(x).view(1, 1, DIN).expand(E, 1, DIN)
    with torch.no_grad():
        out = forward(xt, params)
    return np.asarray(state) + out.mean(dim=0).numpy()[0]


def prediction_mse(params, S, A, S2):
    """Held-out one-step MSE in next-state space, using the ensemble mean."""
    X, Y = to_tensors(S, A, S2, E=params[0].shape[0])
    with torch.no_grad():
        pred = forward(X, params).mean(dim=0)      # ensemble mean delta
    return float(((pred - Y[0]) ** 2).mean())
