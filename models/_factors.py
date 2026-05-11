import numpy as np
from scipy.linalg import qr, svd


def integrator(tilde_Ut, S, tilde_Vt, delta_U, delta_V):
    Ut = tilde_Ut.copy()
    Vt = tilde_Vt.copy()
    K1 = Ut @ S + delta_U.dot(delta_V.T.dot(Vt))
    tilde_U1, tilde_S1 = qr(K1, mode="economic")
    tilde_S0 = tilde_S1 - tilde_U1.T.dot(delta_U.dot(delta_V.T.dot(Vt)))
    L1 = Vt.dot(tilde_S0.T) + delta_V.dot(delta_U.T.dot(tilde_U1))
    tilde_V1, S1 = qr(L1, mode="economic")
    S1 = S1.T
    return tilde_U1, S1, tilde_V1


def truncated_svd_uvT(U, V, rank):
    Q_U, R_U = qr(U, mode="economic")
    Q_V, R_V = qr(V, mode="economic")

    U_svd, S_svd, V_svd = svd(R_U.dot(R_V.T), full_matrices=False)

    U_svd = U_svd[:, :rank]
    S_svd = S_svd[:rank]
    V_svd = V_svd[:rank, :]

    U_new = Q_U.dot(U_svd)
    V_new = V_svd.dot(Q_V.T)
    return U_new, np.diag(S_svd), V_new.T


def symmetric_factorization_qr(X_bar):
    d, B = X_bar.shape
    Q, R = np.linalg.qr(X_bar)
    T = np.eye(B) + R @ R.T

    M = np.linalg.cholesky(T)

    Y_tB = M - np.eye(B)
    return Y_tB, Q, M



