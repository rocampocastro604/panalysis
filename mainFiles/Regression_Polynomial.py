import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.interpolate import interpolate, CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing, svm
from sklearn.datasets import load_iris
from scipy import *
import numpy
import math

np.set_printoptions(suppress=True)

style.use('fivethirtyeight')

rl = LinearRegression(n_jobs=100)
data = pd.read_csv('Temperature.csv', header=0)
x = data['Year']
y = data['Temperature']
b = 0.0
p1 = np.polyfit(x, y, 5)
X = x[:, np.newaxis]
Y = y[:, np.newaxis]
n = len(X)
u = 4  # grado del polinomio
sx = [0] * (n * 2)
a = [0] * u
sy = 0
sx2 = 0
sx3 = 0
sx4 = 0
syx = [0] * (n * 2 + 1)
syx2 = 0
q = 0.0
rl.fit(X, Y)
ml = rl.coef_[0]
b2 = rl.intercept_
v = 1
i = 0
j = 0
marca = [0] * (n + 1)
vector = [0] * (n + 1)
xs = []
M = np.zeros(shape=(u, u))
MN = np.zeros(shape=(u, u + 1))
MM = np.zeros(shape=(u, u))
D = np.zeros(shape=(u, u, u))
m = [0] * (u)
c = [0] * (u)
aa = []
aan = [0] * (u)
ye = 0
K = [0] * (u - 1)
w = np.zeros(shape=((u), (u)))
indice = 0
a0 = 0

for i in range(n):                                          #this method is in charge of make the functions for the matrix
    for v in range(2 * u):
        sx[v] = sx[v] + (X[i] ** (v))
        syx[v] = syx[v] + (Y[i] * (X[i] ** (v)))
    sy = sy + Y[i]

for t in range(u):                             #assign the variables of the result in Y to an array
    c[t] = syx[t]


cont = 0
for b in range(u):                           #add all the values of the results of X to a temporal matrix
    for t in range(u):
        m[t] = sx[t + cont]
    w[b] = m
    cont = cont + 1

for j in range(u):                          #add the values of the last matrix to the real matrix n*n+1
    for i in range(u):
        MN[i, j] = w[j, i]

for j in range(u + 1):                      #add the values of Y to the real matrix in the last column n+1
    for i in range(u):
        if MN[i, j] == 0:
            MN[i, j] = c[i]


for k in range(u):                                               #this is where we apply gaussian elimination with partial pivoting
    for l in range(k, u):
        if abs(MN[l][k]) > abs(MN[k][k]):                        #here we apply the pivoting
            aa = MN[k].copy()
            MN[k] = MN[l]
            MN[l] = aa
    for i in range(k + 1, u):                                   #we apply the gaussian elimination
        q = MN[i, k] / MN[k, k]
        for j in range(k, u + 1):
            MN[i, j] = MN[i, j] - q * MN[k, j]

    xs = [0 for i in range(u)]
xs[u - 1] = float(MN[u - 1][u] / MN[u - 1][u - 1])
for i in range(u - 1, -1, -1):                                      #we fint the values of B
    z = 0
    for j in range(i + 1, u):
        z = z + float(MN[i][j]) * xs[j]
    xs[i] = float(MN[i][u] - z) / MN[i][i]



for i in range(u):                                        #make the polynom
    ye = ye + (xs[i] * (X ** i))

print(ye)
fig = plt.figure(figsize=(8, 7))
print("el valor de r^2; ", r2_score(y, ye))
plt.scatter(x, y, color='blue')
# plt.scatter(X_number, X_predict, color='green')
plt.plot(x, ye)
# plt.plot(xn, f(xn), color='green')
plt.title('Bitcoin over a period of 5 months', fontsize=16)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Close Price', fontsize=16)
plt.show()
