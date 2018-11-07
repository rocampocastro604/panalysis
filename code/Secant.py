def Secante(x0, x1, tol, N, f):
    print(0, x0, x1)
    i = 1
    h = 0.00001  # delta x to calculate f(x) derivate
    while i <= N:
        x = x1 - (x1 - x0) * f(x1) / (f(x1) - f(x0))

        print(i, x)
        if abs(x - x1) < tol:
            return x
        i = i + 1
        x0 = x1  # redefine x0
        x1 = x  # redefine x1

    print('Method failure after %d iterations' % N)


import math  # just in case

# Example data to run the example

f = lambda x: x ** 3 + 4 * x ** 2 - 10
x0 = 1
x1 = 2
tol = 0.0001
N = 20

x = Secante(x0, x1, tol, N, f)
print()
print('The solution is: ', x)