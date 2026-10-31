"""Gaussian Hidden Markov Model implemented from scratch with numpy only.

Model:  hidden state z_t in {0..K-1}, observation x_t in R^D
  P(z_1 = i)            = pi_i
  P(z_t = j | z_t-1 = i) = A_ij
  p(x_t | z_t = k)      = N(x_t; mu_k, diag(sigma2_k))

All recursions (forward, backward, Viterbi) work in log space so that long
sequences do not underflow; sums of probabilities become log-sum-exp.
"""
from __future__ import annotations

import numpy as np


def logsumexp(a: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Numerically stable log(sum(exp(a))) along `axis`."""
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else out.item()


class GaussianHMM:
    """HMM with diagonal-covariance Gaussian emissions.

    Parameters
    ----------
    n_states : number of hidden states K.
    var_floor : lower bound on every emission variance; stops a state from
        collapsing onto a few near-identical windows (e.g. "still").
    smoothing : pseudo-count added to start/transition counts in supervised
        initialisation, so unseen transitions keep a small non-zero probability.
    tol : Baum-Welch stops when the mean per-window log-likelihood improves by
        less than `tol` between iterations.
    max_iter : safety cap on Baum-Welch iterations only.
    """

    def __init__(self, n_states: int, var_floor: float = 1e-2, smoothing: float = 1.0,
                 tol: float = 1e-4, max_iter: int = 500):
        self.K = n_states
        self.var_floor = var_floor
        self.smoothing = smoothing
        self.tol = tol
        self.max_iter = max_iter
        self.history_: list[float] = []

    # ------------------------------------------------------------------ init
    def init_supervised(self, sequences: list[np.ndarray], labels: list[np.ndarray]) -> "GaussianHMM":
        """Set pi, A, mu, sigma2 by counting on labelled sequences.

        pi  : how often each state starts a sequence (+ smoothing)
        A   : counted label transitions t-1 -> t (+ smoothing), row-normalised
        mu, sigma2 : per-state mean / variance of the observations (floored)
        """
        K = self.K
        start = np.full(K, self.smoothing)
        trans = np.full((K, K), self.smoothing)
        for y in labels:
            start[y[0]] += 1
            np.add.at(trans, (y[:-1], y[1:]), 1)
        self.startprob_ = start / start.sum()
        self.transmat_ = trans / trans.sum(axis=1, keepdims=True)
        X, y = np.vstack(sequences), np.concatenate(labels)
        self.means_ = np.array([X[y == k].mean(axis=0) for k in range(K)])
        self.vars_ = np.maximum(np.array([X[y == k].var(axis=0) for k in range(K)]), self.var_floor)
        return self

    # ------------------------------------------------------------ inference
    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """log p(x_t | z_t = k) for every t and k, shape (T, K)."""
        diff = X[:, None, :] - self.means_[None, :, :]
        return -0.5 * (X.shape[1] * np.log(2 * np.pi)
                       + np.log(self.vars_).sum(axis=1)[None, :]
                       + (diff ** 2 / self.vars_[None, :, :]).sum(axis=2))

    def forward(self, log_B: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward pass. Returns log alpha (T, K) and log p(x_1..T)."""
        T = len(log_B)
        log_A = np.log(self.transmat_)
        la = np.empty((T, self.K))
        la[0] = np.log(self.startprob_) + log_B[0]
        for t in range(1, T):
            la[t] = logsumexp(la[t - 1][:, None] + log_A, axis=0) + log_B[t]
        return la, float(logsumexp(la[-1], axis=0))

    def backward(self, log_B: np.ndarray) -> np.ndarray:
        """Backward pass. Returns log beta (T, K) with beta_T = 1."""
        T = len(log_B)
        log_A = np.log(self.transmat_)
        lb = np.zeros((T, self.K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(log_A + (log_B[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood log p(X) of one sequence."""
        return self.forward(self.log_emission(X))[1]

    def posteriors(self, X: np.ndarray) -> np.ndarray:
        """State posteriors gamma_t(k) = p(z_t = k | X), shape (T, K)."""
        log_B = self.log_emission(X)
        la, ll = self.forward(log_B)
        return np.exp(la + self.backward(log_B) - ll)

    def viterbi(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Most likely state path argmax_z p(z, X) and its log-probability."""
        log_B = self.log_emission(X)
        log_A = np.log(self.transmat_)
        T = len(X)
        delta = np.empty((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = np.log(self.startprob_) + log_B[0]
        for t in range(1, T):
            cand = delta[t - 1][:, None] + log_A  # (from, to)
            psi[t] = np.argmax(cand, axis=0)
            delta[t] = cand[psi[t], np.arange(self.K)] + log_B[t]
        path = np.empty(T, dtype=int)
        path[-1] = np.argmax(delta[-1])
        for t in range(T - 1, 0, -1):
            path[t - 1] = psi[t, path[t]]
        return path, float(delta[-1, path[-1]])

    # -------------------------------------------------------------- learning
    def _e_step(self, X: np.ndarray):
        """Expected sufficient statistics of one sequence: gamma, summed xi, log-lik."""
        log_B = self.log_emission(X)
        la, ll = self.forward(log_B)
        lb = self.backward(log_B)
        gamma = np.exp(la + lb - ll)
        log_xi = (la[:-1, :, None] + np.log(self.transmat_)[None]
                  + (log_B[1:] + lb[1:])[:, None, :] - ll)
        return gamma, np.exp(log_xi).sum(axis=0), ll

    def fit_baum_welch(self, sequences: list[np.ndarray]) -> "GaussianHMM":
        """Unsupervised EM refinement starting from the current parameters.

        Stops when the mean per-window log-likelihood improves by less than
        `tol` (normalising by the number of windows makes `tol` independent of
        dataset size); `max_iter` is only a safety cap. `history_` records the
        mean log-likelihood at every iteration for the convergence plot.
        """
        n_obs = sum(len(X) for X in sequences)
        self.history_ = []
        self.converged_ = False
        for _ in range(self.max_iter):
            g0 = np.zeros(self.K)
            xi = np.zeros((self.K, self.K))
            gsum = np.zeros(self.K)
            gx = np.zeros_like(self.means_)
            gxx = np.zeros_like(self.means_)
            total = 0.0
            for X in sequences:
                gamma, xi_s, ll = self._e_step(X)
                g0 += gamma[0]
                xi += xi_s
                gsum += gamma.sum(axis=0)
                gx += gamma.T @ X
                gxx += gamma.T @ X ** 2
                total += ll
            self.history_.append(total / n_obs)
            if len(self.history_) > 1 and self.history_[-1] - self.history_[-2] < self.tol:
                self.converged_ = True
                break
            # M-step (tiny constants keep log() finite for unused states/transitions)
            self.startprob_ = (g0 + 1e-6) / (g0 + 1e-6).sum()
            self.transmat_ = (xi + 1e-6) / (xi + 1e-6).sum(axis=1, keepdims=True)
            denom = np.maximum(gsum, 1e-10)[:, None]
            self.means_ = gx / denom
            self.vars_ = np.maximum(gxx / denom - self.means_ ** 2, self.var_floor)
        self.n_iter_ = len(self.history_)
        return self
