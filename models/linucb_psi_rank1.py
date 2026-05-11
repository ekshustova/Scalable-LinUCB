import numpy as np
from collections import defaultdict
from math import sqrt
from scipy.linalg import norm

from ._factors import integrator, truncated_svd_uvT


class LinUCBwithPSI_rank1:
    """LinUCB with rank-r dynamical low-rank approximation of the inverse
    covariance, updated by rank-1 increments and the projector-splitting
    integrator (per-arm).

    Parameters
    ----------
    num_arms : int
    d : int
        Context dimension.
    epsilon : float
        Ridge regularization (V_0 = epsilon * I).
    alpha : float
        Exploration coefficient.
    rank : int
        Target rank of the low-rank approximation.
    """

    def __init__(self, num_arms, d=10, epsilon=1.0, alpha=1.0, rank=10):
        self.n_arms = num_arms
        self.d = d
        self.epsilon = epsilon
        self.sqrt_epsilon = 1 / sqrt(epsilon)
        self.alpha = alpha
        self.rank = rank

        self.U = defaultdict(lambda: np.zeros((self.d, 2 * rank), dtype=np.float32))
        self.V = defaultdict(lambda: np.zeros((self.d, 2 * rank), dtype=np.float32))
        self.n_cols = 0

        self.Ut = defaultdict(lambda: None)
        self.St = defaultdict(lambda: None)
        self.Vt = defaultdict(lambda: None)

        self.b = defaultdict(lambda: np.zeros(self.d, dtype=np.float32))
        self.theta = defaultdict(lambda: np.zeros(self.d, dtype=np.float32))

    def _L_matvec(self, vec, arm):
        L0_inv = self.sqrt_epsilon * vec
        if self.n_cols == 0:
            return L0_inv
        U = self.U[arm][:, :self.n_cols]
        V = self.V[arm][:, :self.n_cols]
        return L0_inv - U @ (V.T @ L0_inv)

    def update(self, context, arm, reward):
        self.b[arm] += reward * context

        bar_x = self._L_matvec(context, arm)
        norm_sq = norm(bar_x) ** 2
        if norm_sq < 1e-12:
            self._update_theta(arm)
            return

        alpha_t = (sqrt(1 + norm_sq) - 1) / norm_sq
        beta_t = alpha_t / (1 + alpha_t * norm_sq)

        self._update_factors(bar_x, beta_t, arm)
        self._update_theta(arm)

    def _update_factors(self, bar_x, beta_t, arm):
        delta_u = beta_t * bar_x
        if self.n_cols > 0:
            U = self.U[arm][:, :self.n_cols]
            V = self.V[arm][:, :self.n_cols]
            delta_v = bar_x - V @ (U.T @ bar_x)
        else:
            delta_v = bar_x

        self.U[arm][:, self.n_cols] = delta_u
        self.V[arm][:, self.n_cols] = delta_v
        self.n_cols += 1

        if self.n_cols == 2 * self.rank:
            if self.Ut[arm] is None:
                self.Ut[arm], self.St[arm], self.Vt[arm] = truncated_svd_uvT(
                    self.U[arm][:, :self.rank], self.V[arm][:, :self.rank], self.rank
                )
            delta_U_new = self.U[arm][:, self.rank:self.n_cols]
            delta_V_new = self.V[arm][:, self.rank:self.n_cols]
            self.Ut[arm], self.St[arm], self.Vt[arm] = integrator(
                self.Ut[arm], self.St[arm], self.Vt[arm], delta_U_new, delta_V_new
            )
            self.U[arm][:, :self.rank] = self.Ut[arm] @ self.St[arm]
            self.V[arm][:, :self.rank] = self.Vt[arm]
            self.n_cols = self.rank

    def _update_theta(self, arm):
        b_eps = self.epsilon * self.b[arm]
        if self.n_cols == 0:
            self.theta[arm] = b_eps
            return
        U = self.U[arm][:, :self.n_cols]
        V = self.V[arm][:, :self.n_cols]
        Ub = U.T @ b_eps
        Vb = V.T @ b_eps
        self.theta[arm] = b_eps - V @ Ub - U @ Vb + V @ (U.T @ (U @ Vb))

    def score(self, context, arm):
        mean = float(np.dot(self.theta[arm], context))
        U = self.U[arm][:, :self.n_cols]
        V = self.V[arm][:, :self.n_cols]
        v = V.T @ context
        exp = self.sqrt_epsilon * np.linalg.norm(context - U @ v)
        return mean + self.alpha * exp

    def select_arm(self, contexts):
        scores = [self.score(contexts[a], a) for a in range(self.n_arms)]
        return int(np.argmax(scores))
