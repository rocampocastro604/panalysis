import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.interpolate import interpolate,  CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing, svm
from sklearn.datasets import load_iris
from scipy import *
import time

style.use('fivethirtyeight')


def diferencias_divididas(x, lx, ly):
    y = 0
    for i in range(len(lx) - 1):
        if x >= lx[i] and x <= lx[i + 1]:
            y = (ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i]) * (x - lx[i]) + ly[i]
    return y

print("NEWTON WITH DIVIDED DIFFERENCES")
start_time = time.time()
data = pd.read_csv("Example.csv", header=0)
lx = data["DataX"]
ly = data["DataY"]
X = lx[:, np.newaxis]
Y = ly[:, np.newaxis]
n = 500
ye = [0] * (n+1)
xe = [0] * (n+1)
xi = X[0]
xf = X[-1]
m = (xf - xi)/n
for i in range(n+1):
    xe[i] = xi
    xi = xi + m
for i in range(n+1):
    ye[i] = diferencias_divididas(xe[i], lx, ly)

x = 1.5
y = diferencias_divididas(x, lx, ly)

elapsed_time = time.time() - start_time
print(elapsed_time)
print(y)
plt.scatter(x, y, s=200, color='green')
plt.scatter(lx, ly, color='blue')
plt.plot(xe, ye, color='royalblue')
# plt.scatter(x, y, s=200, color='green')
plt.show()
