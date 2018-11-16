import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import *
import os
import sys
from flask import Flask, render_template, request
import json
#GUI
from tkinter import *

style.use('fivethirtyeight')


def linear_regression(file_name, axisx, axisy):

    rl = LinearRegression(n_jobs=100)
    data = pd.read_csv(file_name, header=0)
    x = data["Year"]
    y = data["Temperature"]
    b = 0.0
    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]
    n = len(X)
    sxA = [0] * len(X)
    syA = [0] * len(X)
    sx = 0
    sy = 0
    sx2 = 0
    sxy = 0
    rl.fit(X, Y)
    m = rl.coef_[0]
    b2 = rl.intercept_
    i = 0
    v = 1

    for i in range(n):
        for v in range(2):    #Polynomial Grade
            sxA[v] = sxA[v] + (X[i] ** (v+1))
            syA[v] = syA[v] + (Y[i] ** (v+1))
        sxy = sxy + (X[i] * Y[i])

    sx = sxA[0]
    sy = syA[0]
    sx2 = sxA[1]

    a = ((sy*sx2)-(sx*sxy))/((n*sx2)-(sx*sx))
    b = ((sxy-((sx*sy)/n))/(sx2-((sx*sx)/n)))

    y_e = a + b * X
    y_p = b2 + m * X


    print("el valor de r^2; ", r2_score(y, y_p))
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_e)
    #plt.plot(x, y_p)
    plt.title('Linear Regression', fontsize=16)
    plt.xlabel('Year', fontsize=16)
    plt.ylabel('Temperature', fontsize=16)
    plt.show()

def onClick():
    
    dataset_entered = dataset_entry.get() + ".csv"
    axisx_entered = axisx_entry.get()
    axisy_entered = axisy_entry.get()

    try:
        linear_regression(dataset_entered,axisx_entered, axisy_entered)
    except:
        "There an error somewhere!"

window = Tk()
window.title("MLNA")
window.configure(background = "#2b2b2b")

dataset_txt = Label(window, text="Enter a file name:", font="none 12 bold")
dataset_txt.grid(row=1, column=0,sticky=W)

dataset_entry = Entry(window, width=30, bg="white")
dataset_entry.grid(row=2, column=0,sticky=W)

axisx_txt = Label(window, text = "Enter a column name from the dataset:", font = "none 12 bold")
axisx_txt.grid(row=3, column=0, sticky=W)

axisx_entry = Entry(window, width=30)
axisx_entry.grid(row=4, column= 0, sticky=W)

axisy_txt = Label(window, text = "Enter another column from the dataset:", font = "none 12 bold")
axisy_txt.grid(row=3, column=3, sticky=W)

axisy_entry = Entry(window, width=30)
axisy_entry.grid(row=4, column= 3, sticky=W)

submit_btn = Button(window, text="SUBMIT", width=6, command=onClick)
submit_btn.grid(row=10, column=2,sticky=W)

window.mainloop()

