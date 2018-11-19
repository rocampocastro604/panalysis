from sympy.matrices.matrices import Matrix
from numa import logger, matrix_input, list_input, float_input
from numa.topics.jacobi import norm_error, list_as_float, check_norms


def gaussseidel(a, b, x, e):

    # (1) Build a new vector os approximation
    x1 = list(x)

    error = e + 1 # Mock up error paramenter only to run the While

    while error > e:
        for i in range(a.rows):
            sum_a = sum_b = 0

            for j in range(i):
                sum_a += a[i, j] * x1[j]

            for j in range(i + 1, a.rows):
                sum_b += a[i, j] * x[j]

            # (3) Perform approximation calculation over x1
            x1[i] = (1 / a[i, i]) * (b[i] - sum_a - sum_b)

        # (4) Subtract the second from the first approximation and give us the noma
        error = norm_error(x1, x)

    return list_as_float(x1)