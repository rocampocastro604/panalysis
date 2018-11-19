import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression       # this is the library for linear regression
from sklearn.metrics import r2_score
from scipy import *

style.use('fivethirtyeight')

rl = LinearRegression(n_jobs=100)                # this is for implementation with libraries
data = pd.read_csv('Example.csv', header=0)  # this is the data set of example
x = data['DataX']                                 # this is the data of X
y = data['DataY']                          # this is the data of Y
X = x[:, np.newaxis]                             # this is the data of X added to an array
Y = y[:, np.newaxis]                             # this is the data of Y added to an array
n = len(X)                                       # this is the length of the Data, how many points
sxA = [0] * len(X)                               # this is just an array to save the results
syA = [0] * len(X)                               # this is just an array to save the results
sx = 0                                           # this is the summation of x
sy = 0                                           # this is the summation of y
sx2 = 0                                          # this is the summation of x^2
sxy = 0                                          # this is the summation of x multiplied by y
rl.fit(X, Y)                                     # this is for implementation with libraries
m = rl.coef_[0]                                  # this is for implementation with libraries
b2 = rl.intercept_                               # this is for implementation with libraries

for i in range(n):
    for v in range(2):  # Polynomial Grade
        sxA[v] = sxA[v] + (X[i] ** (v + 1))
        syA[v] = syA[v] + (Y[i] ** (v + 1))
    sxy = sxy + (X[i] * Y[i])

sx = sxA[0]
sy = syA[0]
sx2 = sxA[1]

a = ((sy * sx2) - (sx * sxy)) / ((n * sx2) - (sx * sx))
b = ((sxy - ((sx * sy) / n)) / (sx2 - ((sx * sx) / n)))

y_e = a + b * X

y_p = b2 + m * X


def square_error(Y_origin, Y_line):
    return sum((Y_line - Y_origin) ** 2)


def coefficient_of_determination(Y_origin, Y_line):
    Y_mean_line = [mean(Y_origin) for Y in Y_origin]
    square_error_regression = square_error(Y_origin, Y_line)
    square_error_y_mean = square_error(Y_origin, Y_mean_line)
    print(square_error_regression, square_error_y_mean)
    return 1 - (square_error_regression / square_error_y_mean)


r_square = coefficient_of_determination(Y, y_p)

predict_x = 1.5
predict_y = (m * predict_x) + a

print("El valor de Y para PredictX es: ", predict_y)
print("el valor de r^2; ", r2_score(y, y_p))
print("el valor de r^2, propio; ", r2_score(y, y_e))
plt.scatter(predict_x, predict_y, s=100, color='green')
plt.scatter(x, y, color='blue')
plt.plot(x, y_p, color='royalblue')
# plt.scatter(predict_x, predict_y, color='green')
# plt.plot(x, y_p)
plt.show()
