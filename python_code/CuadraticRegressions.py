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

style.use('fivethirtyeight')

rl = LinearRegression(n_jobs=100)
data = pd.read_csv('Temperature.csv', header=0)
x = data['Year']
y = data['Temperature']
b = 0.0
X = x[:, np.newaxis]
Y = y[:, np.newaxis]
p1 = np.polyfit(x, y, 3)
cs = CubicSpline(x, y)
n = len(X)
sx = [0] * len(X)
sy = 0
sx2 = 0
sx3 = 0
sx4 = 0
syx = [0] * len(X)
syx2 = 0
rl.fit(X, Y)
ml = rl.coef_[0]
b2 = rl.intercept_
v = 1
i = 0
j = 0
M = np.zeros(shape=(3, 3))
M0 = np.zeros(shape=(3, 3))
M1 = np.zeros(shape=(3, 3))
M2 = np.zeros(shape=(3, 3))
m = [0] * 3
N = [0] * 3
P = [0] * 3
c = [0] * 3
u = 4  #grado del polinomio

for i in range(n):
    for v in range(u):    #Polynomial Grade
        sx[v] = sx[v] + (X[i] ** (v+1))
        syx[v] = syx[v] + (Y[i] * (X[i] ** (v+1)))
    sy = sy + Y[i]

#for i in range(9):
#    for v in range(n):    #Polynomial Grade
#        sx[i] = sx + (X[i] ** v)
#        sx2 = sx2 + (X[i] ** 2)
#        sx3 = sx3 + (X[i] ** 3)
#        sx4 = sx4 + (X[i] ** 4)
#        sy = sy + Y[i]
#        syx = syx + (Y[i] * X[i])
#        syx2 = syx2 + (Y[i] * (X[i] ** 2))

for t in range(u):
    if c[t] == c[0]:
        c[t] = sy
    elif c[t] != c[0]:
        c[t] = syx[t-1]

for t in range(u-1):
    if m[t] == m[0]:
        m[t] = n
    elif m[t] != m[0]:
        m[t] = sx[t-1]

for t in range(u-1):
    N[t] = sx[t]

for t in range(u-1):
    P[t] = sx[t+1]

for j in range(u-1):
    for i in range(u-1):
        if j == 0:
            M[i, j] = m[i]
        if j == 1:
            M[i, j] = N[i]
        if j == 2:
            M[i, j] = P[i]

for j in range(u-1):
    for i in range(u-1):
        if j == 0:
            M0[i, j] = c[i]
        if j == 1:
            M0[i, j] = N[i]
        if j == 2:
            M0[i, j] = P[i]

for j in range(u-1):
    for i in range(u-1):
        if j == 0:
            M1[i, j] = m[i]
        if j == 1:
            M1[i, j] = c[i]
        if j == 2:
            M1[i, j] = P[i]

for j in range(u-1):
    for i in range(u-1):
        if j == 0:
            M2[i, j] = m[i]
        if j == 1:
            M2[i, j] = N[i]
        if j == 2:
            M2[i, j] = c[i]

a0 = np.linalg.det(M0)/np.linalg.det(M)
a1 = np.linalg.det(M1)/np.linalg.det(M)
a2 = np.linalg.det(M2)/np.linalg.det(M)

print(a0)
print(a1)
print(a2)

y_e = a0 + (a1 * X) + (a2 * (X ** 2))

#a = ((sy*sx2)-(sx*sxy))/((n*sx2)-(sx*sx))
#b = ((sxy-((sx*sy)/n))/(sx2-((sx*sx)/n)))

#y_e = B0 + B1 * X + B2 * (X*X)


print("el valor de r^2; ", r2_score(y, y_e))
plt.scatter(x, y, color='blue')
# plt.scatter(X_number, X_predict, color='green')
#plt.plot(x, y_e)
plt.plot(X, np.polyval(p1, X))
#plt.plot(xn, f(xn), color='green')
plt.title('Regresion lineal', fontsize=16)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Temperature', fontsize=16)
plt.show()