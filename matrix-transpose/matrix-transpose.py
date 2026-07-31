import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    n, m = len(A), len(A[0])
    A_T = np.empty((m, n), dtype=float)
    for i in range(n):
        for j in range(m):
            A_T[j, i] = A[i][j]
    return A_T
