import numpy as np
import scipy.linalg


def mat_sqrt_cholesky(M: np.ndarray):
    """
    Evaluate the matrix square root L via cholesky decomposition:
     - M = L L^T
    """

    if not np.allclose(M, M.T):
        raise ValueError("Matrix is not square symmetric.")

    L = scipy.linalg.cholesky(M, lower=True)

    if not np.isreal(L).all():
        raise ValueError("Matrix is not positive semi-definite")

    assert np.allclose(L @ L.T, M)
    return L



def mat_sqrt_eigen(M: np.ndarray):
    """
    Evaluate the matrix square root A via eigen-decomposition:
     - M = V D V^T
     - A = V D^(1/2)
    """

    if not (M == M.T).all():
        raise ValueError("Matrix is not square square symmetric.")

    values, vectors = scipy.linalg.eigh(M)
    D = np.diag(values)
    V = vectors

    A = V @ np.sqrt(D)

    assert np.allclose(A @ A.T, M)
    return A


def mat_sqrt_eigen_preserve_vectors(M: np.ndarray):
    """
    Evaluate the matrix square root S via eigen-decomposition s.t. eigenvectors are preserved:
     - M = V D V^T
     - S = V D^(1/2) V^(-1)
    """

    if not (M == M.T).all():
        raise ValueError("Matrix is not square square symmetric.")

    values, vectors = scipy.linalg.eigh(M)
    D = np.diag(values)
    V = vectors

    S = V @ np.sqrt(D) @ np.linalg.inv(V)

    assert np.allclose(S @ S.T, M)
    return S


# TESTING
if __name__ == "__main__":
    M = np.array([
        [1.1335, 1.9544],
        [1.9544, 5.5336]
    ])

    L = mat_sqrt_cholesky(M)
    print("cholesky:\n", L)

    A = mat_sqrt_eigen(M)
    print("eigen:\n", A)

    S = mat_sqrt_eigen_preserve_vectors(M)
    print("eigen_preserve_vectors:\n", S)
