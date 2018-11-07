#Newton method
def dx(f, x):
    return abs(0-f(x))
 
def newton(f, df, x0, e):
    delta = dx(f, x0)
    while delta > e:
        x0 = x0 - f(x0)/df(x0)
        delta = dx(f, x0)
    print ('Root is at: ', x0)
    print ('f(x) at root is: ', f(x0))

def f(x):
    return 3*x**3 + 2*x**2

def df(x):
    return 9*x**2 + 4*x

def main():
    x = newton(f, df, 0, 1e-5)
    print(x, f(x))

newton(f, df, 0.1, 54 )