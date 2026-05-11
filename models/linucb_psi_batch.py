import numpy as np
from collections import defaultdict
from numpy.linalg import multi_dot

from ._factors import integrator, truncated_svd_uvT, symmetric_factorization_qr


class LinUCBwithPSI_Batch:
    """Batched LinUCB-PSI: rank-B updates via symmetric factorization +
    projector-splitting integrator (per-arm).

    A batch of B contexts is consumed at once instead of B rank-1 steps,
    which is faster for large B.

    Parameters
    ----------
    num_arms : int
    d : int
    epsilon : float
        Ridge regularization (V_0 = epsilon * I).
    alpha : float
        Exploration coefficient.
    rank : int
        Target rank of the low-rank approximation.
    """

    def __init__(self, num_arms, d, epsilon=1.0, alpha=1.0, rank=10):
        self.num_arms = num_arms
        self.d = d
        self.epsilon = epsilon
        self.eps = 1 / np.sqrt(epsilon)
        self.alpha = alpha
        self.rank = rank

        self.U = defaultdict(lambda: np.empty((self.d, 0), dtype=np.float32))
        self.V = defaultdict(lambda: np.empty((self.d, 0), dtype=np.float32))
        self.U_psi = defaultdict(lambda: None)
        self.S_psi = defaultdict(lambda: None)
        self.V_psi = defaultdict(lambda: None)

        self.b = defaultdict(lambda: np.zeros(self.d, dtype=np.float32))
        self.theta = defaultdict(lambda: np.zeros(self.d, dtype=np.float32))

    def update(self, X_batch, arm, rewards):
        """Apply a single batch update.

        X_batch : (d, B) array, columns are contexts.
        rewards : (B,) array.
        """
        self.b[arm] += X_batch @ rewards

        if self.U[arm].shape[1] > 0:
            X_bar = (X_batch - multi_dot([self.U[arm], self.V[arm].T, X_batch])) * self.eps
        else:
            X_bar = X_batch * self.eps

        try:
            Y_tB, Q, M = symmetric_factorization_qr(X_bar)
            from scipy.linalg import solve_triangular
            M_inv = solve_triangular(M, np.eye(M.shape[0]), lower=True)
            C = Y_tB @ M_inv
        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error for arm {arm}: {e}")
            return False

        self._apply_psi_batch(arm, Q, C)
        self._update_theta(arm)
        return True

    def _apply_psi_batch(self, arm, Q, C):
        U_update = Q @ C
        if self.V[arm].shape[1] > 0:
            V_update = Q - multi_dot([self.V[arm], self.U[arm].T, Q])
        else:
            V_update = Q

        current_cols = self.U[arm].shape[1]

        if current_cols < self.rank:
            self.U[arm] = (np.column_stack([self.U[arm], U_update])
                           if current_cols > 0 else U_update)
            self.V[arm] = (np.column_stack([self.V[arm], V_update])
                           if current_cols > 0 else V_update)
            return

        if self.U_psi[arm] is None:
            self.U_psi[arm], self.S_psi[arm], self.V_psi[arm] = truncated_svd_uvT(
                self.U[arm], self.V[arm], self.rank
            )

        self.U_psi[arm], self.S_psi[arm], self.V_psi[arm] = integrator(
            self.U_psi[arm], self.S_psi[arm], self.V_psi[arm], U_update, V_update
        )
        self.U[arm] = self.U_psi[arm] @ self.S_psi[arm]
        self.V[arm] = self.V_psi[arm]

    def _update_theta(self, arm):
        b_eps = (self.eps ** 2) * self.b[arm]
        if self.U[arm].shape[1] == 0:
            self.theta[arm] = b_eps
            return
        Ub = self.U[arm].T @ b_eps
        Vb = self.V[arm].T @ b_eps
        self.theta[arm] = (b_eps
                           - self.V[arm] @ Ub
                           - self.U[arm] @ Vb
                           + self.V[arm] @ (self.U[arm].T @ (self.U[arm] @ Vb)))

    def score(self, context, arm):
        mean = float(np.dot(self.theta[arm], context))
        if self.U[arm].shape[1] == 0:
            exp = self.eps * np.linalg.norm(context)
        else:
            v = self.V[arm].T @ context
            exp = self.eps * np.linalg.norm(context - self.U[arm] @ v)
        return mean + self.alpha * exp

    def select_arm(self, contexts):
        scores = [self.score(contexts[a], a) for a in range(self.num_arms)]
        return int(np.argmax(scores))
