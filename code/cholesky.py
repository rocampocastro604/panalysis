import math


def Cholesky(A, b):
    n = len(A)

    # (1) Create L with zeros
    L = [[0 for i in range(n)] for i in range(n)]

    # (2) Fullfill L
    for k in range(0, n, 1):  # Column by Column
        # Discover the pivot for that column
        for i in range(0, k, 1):
            L[k][k] -= (L[k][i]) ** 2

        L[k][k] = math.sqrt(A[k][k] - L[k][k])

        for i in range(k + 1, n, 1):
            for j in range(0, k, 1):
                L[i][k] -= L[i][j] * L[k][j]
            L[i][k] = (A[k][i] - L[i][k]) / float(L[k][k])

    Lt = [list(i) for i in zip(*L)]

    # (5) Perform substitutioan Ly=b
    y = [0 for i in range(n)]
    for i in range(0, n, 1):
        y[i] = b[i] / float(L[i][i])
        for k in range(0, i, 1):
            y[i] -= y[k] * L[i][k]

    # (6) Perform substitution Ltx=y
    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = y[i] / float(Lt[i][i])
        for k in range(i - 1, -1, -1):
            x[i] -= x[i] * Lt[i][k]

    return x