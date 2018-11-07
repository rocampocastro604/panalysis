import cmath, math, os, sys;

#Not working with some functions

def main():
    fixedpoint();


def fixedpoint():
    x = float(input('Enter initial value(X0): '));
    tolerancia = float(input('Enter error percentage %: '));
    N = int(input('Max number of iterations: '));
    f = input('Enter f(x) function: ');
    er = 100;
    i = 0;
    print('#iteration\tg(f(x))\t\terror')
    while (i <= N and er >= tolerancia):
        temp = x;
        x = eval(f);
        er = abs((x - temp));
        print("%d\t\t%.4f\t\t%.4f" % (i, x, er));
        i += 1;

    print("\nThe closes solution is: %.4f with an error of %.4f" % (x, er));


if __name__ == "__main__":
    main();