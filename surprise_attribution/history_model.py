"""Forward model with a short history window (see history_config.py)."""
import os

import numpy as np
import torch
import torch.nn as nn

import compliant_trials as ct
import config
import history_config as hc

HERE = os.path.dirname(os.path.abspath(__file__))


class HistoryModel(nn.Module):
    def __init__(self, k, hidden=config.HIDDEN):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(nn.Linear(4 * k, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 3))

    def forward(self, s_hist, a_hist):
        """s_hist (N, k, 3), a_hist (N, k, 1) -> predicted delta (N, 3)."""
        n = s_hist.shape[0]
        return self.net(torch.cat([s_hist.reshape(n, -1), a_hist.reshape(n, -1)], dim=1))

    def predict_next(self, s_hist, a_hist):
        return s_hist[:, -1, :] + self.forward(s_hist, a_hist)


def windows(S, A, S2, k, segment=200):
    """History windows from sequential data that resets every `segment` steps.

    Target i is used only if the k-1 previous transitions are in the same segment.
    Returns (s_hist, a_hist, s_next, index_of_target).
    """
    idx = np.array([i for i in range(len(S)) if i % segment >= k - 1])
    sh = np.stack([S[i - k + 1:i + 1] for i in idx])
    ah = np.stack([A[i - k + 1:i + 1] for i in idx])
    return sh.astype(np.float32), ah.astype(np.float32), S2[idx], idx


def train(sh, ah, s2, k, seed=config.SEED, epochs=config.TRAIN_EPOCHS):
    torch.manual_seed(seed)
    m = HistoryModel(k)
    sh_t, ah_t, s2_t = map(torch.from_numpy, (sh, ah, s2))
    target = s2_t - sh_t[:, -1, :]
    opt = torch.optim.Adam(m.parameters(), lr=config.LEARNING_RATE)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        loss = loss_fn(m(sh_t, ah_t), target)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m, float(loss.item())


def get(name, cfg, k):
    """Train (or load cached) the K-history model for one configuration."""
    d = os.path.join(HERE, hc.MODEL_DIR); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}_k{k}.pt")
    if os.path.exists(path):
        ck = torch.load(path)
        m = HistoryModel(k); m.load_state_dict(ck["state_dict"]); m.eval()
        return m, ck["train_mse"]
    torch.set_num_threads(hc.TRAIN_THREADS)
    S, A, S2 = ct.collect(cfg, ct.TRAIN_N, config.SEED)      # same data and seed as before
    sh, ah, s2, _ = windows(S, A, S2, k)
    m, mse = train(sh, ah, s2, k)
    torch.save({"state_dict": m.state_dict(), "k": k, "train_mse": mse, "n_train": len(s2)}, path)
    return m, mse


def heldout(single, hist_models, cfg):
    """Held-out one-step error on the SAME 1000 fresh transitions as before.

    Every model is scored on the same targets: those with a full history for the
    largest k (so the single-state number here is on a slightly smaller subset
    than results_compliant's 'heldout_error', which is also reported).
    """
    S, A, S2 = ct.collect(cfg, ct.HELDOUT_N, config.SEED + 1)
    kmax = max(hist_models)
    _, _, s2, idx = windows(S, A, S2, kmax)
    out = {}
    with torch.no_grad():
        p = single.predict_next(torch.from_numpy(S[idx]), torch.from_numpy(A[idx])).numpy()
        out["single"] = float(np.linalg.norm(p - s2, axis=1).mean())
        for k, m in hist_models.items():
            sh = np.stack([S[i - k + 1:i + 1] for i in idx]); ah = np.stack([A[i - k + 1:i + 1] for i in idx])
            p = m.predict_next(torch.from_numpy(sh), torch.from_numpy(ah)).numpy()
            out[f"k{k}"] = float(np.linalg.norm(p - s2, axis=1).mean())
    out["n_targets"] = int(len(idx))
    return out


def episode_errors(m, obs, act):
    """Per-step error of a history model along a recorded episode.

    obs: (T+1, 3) observed states s_0..s_T; act: (T, 1). err[t] scores the
    prediction of s_{t+1}; steps with t < k-1 (no full history) are NaN. The
    rule never reads them: the earliest onset is step 55.
    """
    k, T = m.k, act.shape[0]
    err = np.full(T, np.nan, dtype=np.float32)
    ts = np.arange(k - 1, T)
    sh = np.stack([obs[t - k + 1:t + 1] for t in ts]).astype(np.float32)
    ah = np.stack([act[t - k + 1:t + 1] for t in ts]).astype(np.float32)
    with torch.no_grad():
        p = m.predict_next(torch.from_numpy(sh), torch.from_numpy(ah)).numpy()
    err[ts] = np.linalg.norm(p - obs[ts + 1], axis=1)
    return err
