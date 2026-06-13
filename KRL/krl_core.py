"""
KRL core: kernelized rank learning via BMRM.

Port of https://github.com/BorgwardtLab/Kernelized-Rank-Learning (He et al. 2018)
from Python 2 to Python 3.  Changes: xrange → range, print() → print, cvxopt
imports streamlined, no-op function removed for Python 2's ``\n``.

Interface
---------
>>> from krl_core import KRL
>>> W = KRL_fit(X_train, Y_train, k=10, Lambda=0.001, gamma=0.001, njobs=4)
>>> Y_pred = KRL_pred(W, X_train, X_test, gamma=0.001)
"""
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import rbf_kernel
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


# ---------------------------------------------------------------------------
# Loss / gradient
# ---------------------------------------------------------------------------

def ndcgk_vector_loss_gradient(y, f, k):
    """NDCG@k loss + gradient for a single cell line (listwise ranking)."""
    if y.shape[0] < k:
        k = y.shape[0]
    m = len(y)
    a = np.zeros(m)
    a[:k] = 1.0 / np.log(np.arange(2, k + 2))
    b = 2 ** y - 1
    c = np.arange(1, m + 1) ** (-0.25)
    C = np.outer(a, b) - np.outer(c, f)
    pi = linear_sum_assignment(C)[1]          # optimal permutation
    loss = np.dot(f[pi] - f, c) + np.dot(a - a[pi], b)
    pi_inverse = np.argsort(pi)
    gradient = c[pi_inverse] - c
    return loss, gradient


def ndcgk_block_loss_gradient(index, F, Y, k, Notnan):
    n = Y.shape[0]
    l = 0
    g = np.zeros(F.shape)
    for i in range(n):
        f, y = F[i, Notnan[i]], Y[i, Notnan[i]]
        order = np.argsort(y)[::-1]
        back_order = np.argsort(order)
        loss, gradient = ndcgk_vector_loss_gradient(y[order], f[order], k)
        l += loss
        g[i, Notnan[i]] = gradient[back_order]
    return index, l, g


def ndcgk_row_loss_gradient(W, X, Y, k, Notnan, njobs):
    n, m = Y.shape
    p = X.shape[1]
    W = W.reshape(p, m)
    l = 0
    F = np.dot(X, W)
    g = np.zeros(F.shape)

    step = max(1, n // njobs)
    indices = list(range(0, n, step))
    results = Parallel(n_jobs=njobs)(
        delayed(ndcgk_block_loss_gradient)(i, F[i:i + step], Y[i:i + step], k,
                                           Notnan[i:i + step])
        for i in indices)
    for idx, li, gi in results:
        l += li
        g[idx:idx + step] = gi

    g = np.dot(X.T, g)
    return l, g


def ndcgk_block_loss(F, Y, k, Notnan):
    n = Y.shape[0]
    l = 0
    for i in range(n):
        f, y = F[i, Notnan[i]], Y[i, Notnan[i]]
        order = np.argsort(y)[::-1]
        loss, _ = ndcgk_vector_loss_gradient(y[order], f[order], k)
        l += loss
    return l


def ndcgk_row_loss(W, X, Y, k, Notnan, njobs):
    n, m = Y.shape
    p = X.shape[1]
    W = W.reshape(p, m)
    F = np.dot(X, W)

    step = max(1, n // njobs)
    indices = list(range(0, n, step))
    results = Parallel(n_jobs=njobs)(
        delayed(ndcgk_block_loss)(F[i:i + step], Y[i:i + step], k,
                                  Notnan[i:i + step])
        for i in indices)
    return sum(results)


# ---------------------------------------------------------------------------
# BMRM (Bundle Method for Regularised Risk Minimisation)
# ---------------------------------------------------------------------------

def solve_Wt_kernel(A, b, t, Lambda, K_inv):
    """Solve the inner QP of the BMRM algorithm (lines 4-5 in Algo 1 of
    Teo et al. 2010, JMLR).  Returns W_t."""
    A1 = A[1:t + 1]
    n, m = A1.shape[1], A1.shape[2]
    P = np.zeros((t, t))
    for i in range(m):
        Ai = A1[:, :, i]                        # (t, n)
        P += Ai @ K_inv @ Ai.T
    P = matrix(P / Lambda)
    q = -matrix(b[1:t + 1])
    G = matrix(-np.eye(t))
    h = matrix(np.zeros(t))
    AA = matrix(1.0, (1, t))
    bb = matrix(1.0)
    sol = solvers.qp(P, q, G, h, AA, bb)
    alphat = np.ravel(np.array(sol['x']))        # (t,)
    u = -np.einsum('tnm,t->nm', A1, alphat) / Lambda   # (n, m)
    Wt = np.zeros((n, m))
    for i in range(m):
        Wt[:, i] = np.dot(u[:, i], K_inv)        # K_inv (n, n)
    return Wt


def bmrm_kernel(W0, loss, loss_gradient, args, Lambda, K, K_inv,
                FTOL=1e-4, MAX_ITER=1000, verbose=True):
    """BMRM with a kernelised objective.

    Parameters
    ----------
    W0 : ndarray (n, m)
        Initial weight matrix.
    loss : callable
        ``loss(W, *args) → float``
    loss_gradient : callable
        ``loss_gradient(W, *args) → (loss, gradient)``  gradient shape (n, m).
    args : tuple
        Extra args passed to loss / loss_gradient.
    Lambda : float
        Regularisation parameter.
    K : ndarray (n, n)
        Kernel matrix (precomputed).
    K_inv : ndarray (n, n)
        Inverse of the kernel matrix.
    FTOL : float
        Stopping tolerance (dual gap).
    MAX_ITER : int
        Maximum BMRM iterations.
    verbose : bool
        Print progress.
    """
    n, m = W0.shape
    W_store = np.zeros((MAX_ITER, n, m))
    W_store[0] = W0
    A = np.zeros((MAX_ITER + 1, n, m))
    b_arr = np.zeros(MAX_ITER + 1)

    b_arr[1], A[1] = loss_gradient(W0, *args)
    reg = 0.5 * Lambda * (W0.T @ K @ W0).trace()
    fval = np.zeros(MAX_ITER + 1)
    fval[0] = b_arr[1] + reg
    b_arr[1] = b_arr[1] - np.multiply(A[1], W0).sum()

    for t in range(1, MAX_ITER):
        W_store[t] = solve_Wt_kernel(A, b_arr, t, Lambda, K_inv)
        b_arr[t + 1], A[t + 1] = loss_gradient(W_store[t], *args)
        reg = 0.5 * Lambda * (W_store[t].T @ K @ W_store[t]).trace()
        fval[t] = b_arr[t + 1] + reg
        b_arr[t + 1] = b_arr[t + 1] - np.multiply(A[t + 1], W_store[t]).sum()

        fval_lb_t = np.multiply(A[t], W_store[t]).sum() + b_arr[t] + reg
        epsilon = (fval[:t + 1] - fval_lb_t).min()
        if verbose:
            print("BMRM iter {}: epsilon={:.6g}".format(t, epsilon))
        if epsilon < FTOL:
            break

    return W_store[t]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def KRL_fit(X, Y, k, Lambda, gamma, njobs=1, verbose=True):
    """Fit a KRL model.

    Parameters
    ----------
    X : ndarray (n_cells, n_genes)
        Training cell line features (raw gene expression).
    Y : ndarray (n_cells, n_drugs)
        Training drug sensitivity (-log IC50), NaN for unmeasured.
    k : int
        NDCG@k truncation used during training.
    Lambda : float
        Regularisation strength.
    gamma : float
        RBF kernel width.
    njobs : int
        Number of parallel workers.
    verbose : bool

    Returns
    -------
    W : ndarray (n_cells, n_drugs)
        Weight matrix (in the kernelised feature space).
    """
    K = rbf_kernel(X, gamma=gamma)
    K_inv = np.linalg.inv(K + 1e-6 * np.eye(K.shape[0]))

    n, m = K.shape[0], Y.shape[1]
    Notnan = [~np.isnan(Y[i]) for i in range(n)]

    np.random.seed(0)
    W0 = 0.01 * np.random.randn(n, m)
    W = bmrm_kernel(W0, ndcgk_row_loss, ndcgk_row_loss_gradient,
                    (K, Y, k, Notnan, njobs), Lambda, K, K_inv,
                    MAX_ITER=1000, verbose=verbose)
    return W.reshape(n, m)


def KRL_pred(W, X_train, X_test, gamma):
    """Predict drug sensitivities for test cell lines.

    Parameters
    ----------
    W : ndarray (n_train, n_drugs)
        Weight matrix from KRL_fit.
    X_train : ndarray (n_train, n_genes)
    X_test : ndarray (n_test, n_genes)
    gamma : float

    Returns
    -------
    Y_pred : ndarray (n_test, n_drugs)
    """
    K = rbf_kernel(X_test, X_train, gamma=gamma)
    return K @ W


def KRL(X_train, Y_train, X_test, k, Lambda, gamma, njobs=1, verbose=True):
    """Convenience: fit + predict."""
    W = KRL_fit(X_train, Y_train, k, Lambda, gamma, njobs, verbose)
    return KRL_pred(W, X_train, X_test, gamma)
