import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import *
import time
#GUI
#from tkinter import *

style.use('fivethirtyeight')

def main():
	
	print("----------------------")
	print("WELCOME TO MLNA")
	print("----------------------")
	print("Select an option: ")
	print("1. Linear Regression")
	print("2. Polynomial Regression")
	print("3. Newton with divided differences")
	print("4. Lagrange Method")
	print("5. Compare Newton with Lagrange")
	lr_pr = int(input("> "))

	
	if(lr_pr == 1):
		print("//////////LINEAR REGRESSION//////////")
		the_file=str(input("Choose a Dataset: ")) + ".csv"
		the_axisx=str(input("Choose a variable for the X axis: "))
		the_axisy=str(input("Choose a variable for the Y axis: "))
		linear_regression(the_file, the_axisx, the_axisy)
	elif(lr_pr == 2):
		print("//////////POLYNOMIAL REGRESSION//////////")
		the_file=str(input("Choose a Dataset: ")) + ".csv"
		the_axisx=str(input("Choose a variable for the X axis: "))
		the_axisy=str(input("Choose a variable for the Y axis: "))
		the_grade=int(input("Choose a grade for the polynomn: "))
		polynomial_regression(the_file, the_axisx, the_axisy, the_grade)
	elif(lr_pr == 3):
		print("//////////NEWTON DIVIDED DIFERENCES//////////")
		the_file=str(input("Choose a Dataset: ")) + ".csv"
		the_axisx=str(input("Choose a variable for the X axis: "))
		the_axisy=str(input("Choose a variable for the Y axis: "))
		newton_dd(the_file, the_axisx, the_axisy)
	elif(lr_pr == 4):
		print("//////////LAGRANGE METHOD//////////")
		the_file=str(input("Choose a Dataset: ")) + ".csv"
		the_axisx=str(input("Choose a variable for the X axis: "))
		the_axisy=str(input("Choose a variable for the Y axis: "))
		lagrange(the_file, the_axisx, the_axisy)
	elif(lr_pr == 5):
		the_file=str(input("Choose a Dataset: ")) + ".csv"
		the_axisx=str(input("Choose a variable for the X axis: "))
		the_axisy=str(input("Choose a variable for the Y axis: "))
		newton_dd(the_file, the_axisx, the_axisy)
		lagrange(the_file, the_axisx, the_axisy)
	else:
		print("Not defined option")


def polynomial_regression(file_name, axisx, axisy, grade):

	np.set_printoptions(suppress=True)

	style.use('fivethirtyeight')

	rl = LinearRegression(n_jobs=100)
	data = pd.read_csv(file_name, header=0)
	x = data[axisx]
	y = data[axisy]
	b = 0.0
	p1 = np.polyfit(x, y, grade)
	X = x[:, np.newaxis]
	Y = y[:, np.newaxis]
	#cs = CubicSpline(x, y)
	n = len(X)
	u = grade #polynomial grade
	sx = [0] * (n*2)
	a = [0] * u
	sy = 0
	sx2 = 0
	sx3 = 0
	sx4 = 0
	syx = [0] * (n*2+1)
	syx2 = 0
	rl.fit(X, Y)
	ml = rl.coef_[0]
	b2 = rl.intercept_
	v = 1
	i = 0
	j = 0
	marca = [0] * (n+1)
	vector = [0] * (n+1)
	M = np.zeros(shape=(u, u))
	MN = np.zeros(shape=(u, u+1))
	MM = np.zeros(shape=(u, u))
	D = np.zeros(shape=(u, u, u))
	m = [0] * (u)
	c = [0] * (u)
	ye = 0
	K = [0] * (u-1)
	w = np.zeros(shape=((u), (u)))
	indice = 0
	a0 = 0

	for i in range(n):
	    for v in range(2*u):
	        sx[v] = sx[v] + (X[i] ** (v))
	        syx[v] = syx[v] + (Y[i] * (X[i] ** (v)))
	    sy = sy + Y[i]


	for t in range(u):
	    c[t] = syx[t]

	cont = 0
	for b in range(u):
	    for t in range(u):
	        m[t] = sx[t + cont]
	    w[b] = m
	    cont = cont + 1


	for j in range(u):
	    for i in range(u):
	        M[i, j] = w[j, i]
	        MN[i, j] = w[j, i]

	for j in range(u + 1):
	    for i in range(u):
	        if MN[i, j] == 0:
	            MN[i, j] = c[i]


	for k in range(u):
	    for i in range(k+1, u):
	        indice = MN[i, k]/MN[k, k]
	        for j in range(k, u+1):
	            MN[i, j] = MN[i, j] - indice * MN[k, j]

	xs = [0 for i in range(u)]


	xs[u-1] = float(MN[u-1][u])/MN[u-1][u-1]
	for i in range (u-1,-1,-1):
	    z = 0
	    for j in range(i+1,u):
	        z = z  + float(MN[i][j])*xs[j]
	    xs[i] = float(MN[i][u] - z)/MN[i][i]


	'''for i in range(u, -1, -1):
	    for j in range(u):
	        a[j] ='''


	for k in range(u):
	    for j in range(u):
	        for i in range(u):
	            if j == k:
	                MM[i, j] = c[i]
	            elif j != k:
	                MM[i, j] = w[j, i]
	    D[k] = MM



	for i in range(u):
	    a[i] = np.linalg.det(D[i])/np.linalg.det(M)


	for i in range(u):
	    print(xs[i])
	    ye = ye + (xs[i] * (X ** i))


	#a = ((sy*sx2)-(sx*sxy))/((n*sx2)-(sx*sx))
	#b = ((sxy-((sx*sy)/n))/(sx2-((sx*sx)/n)))

	#y_e = B0 + B1 * X + B2 * (X*X)


	fig = plt.figure(figsize=(8, 7))
	print("r^2 value is: ", r2_score(y, ye))
	plt.scatter(x, y, color='blue')
	# plt.scatter(X_number, X_predict, color='green')
	plt.plot(x, ye)
	#plt.plot(xn, f(xn), color='green')
	plt.title('Polynomial Regression' , fontsize=16)
	plt.xlabel(axisx, fontsize=16)
	plt.ylabel(axisy, fontsize=16)
	plt.show()

def linear_regression(file_name, axisx, axisy):

    rl = LinearRegression(n_jobs=100)
    data = pd.read_csv(file_name, header=0)
    x = data[axisx]
    y = data[axisy]
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


    print("r^2 value: ", r2_score(y, y_p))
    plt.scatter(x, y, color='blue')
    plt.plot(x, y_e)
    #plt.plot(x, y_p)
    plt.title('Linear Regression', fontsize=16)
    plt.xlabel(axisx, fontsize=16)
    plt.ylabel(axisy, fontsize=16)
    plt.show()

def newton_dd(file_name, axisx, axisy):
	
	def divdif(x, lx, ly):
		y = 0
		for i in range(len(lx) - 1):
			if x >= lx[i] and x <= lx[i + 1]:
				y = (ly[i + 1] - ly[i]) / (lx[i + 1] - lx[i]) * (x - lx[i]) + ly[i]
		return y
	
	def main_dd():
		start_time = time.time()
		data = pd.read_csv(file_name, header=0)
		lx = data[axisx]
		ly = data[axisy]
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
			ye[i] = divdif(xe[i], lx, ly)

		x = float(input("The value to interpolate: "))
		y = divdif(x, lx, ly)

		elapsed_time = time.time() - start_time
		print("It took ",elapsed_time, " Seconds")
		print("The y value for the entered x is: ", y)
		plt.scatter(x, y, s=200, color='green')
		plt.scatter(lx, ly, color='blue')
		plt.plot(xe, ye, color='royalblue')
		plt.title('Newton Divided Differences', fontsize=18)
    	plt.xlabel(axisx, fontsize=16)
    	plt.ylabel(axisy, fontsize=16)
		# plt.scatter(x, y, s=200, color='green')
		plt.show()
	main_dd()


def lagrange(file_name, axisx, axisy):
	
	def lagrange_son(x, lx, ly):
		y = 0
		for i in range(len(lx)):
			L = 1
			for j in range(len(lx)):
				if  j != i:
					L = L*(x-lx[j])/(lx[i]-lx[j])
			y = y + L*ly[i]
		return y


	def main_lg():
		start_time = time.time()
		data = pd.read_csv(file_name, header=0)
		lx = data[axisx]
		ly = data[axisy]
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
			ye[i] = lagrange_son(xe[i], lx, ly)

		x = float(input("The value to interpolate: "))
		y = lagrange_son(x, lx, ly)

		elapsed_time = time.time() - start_time
		print("It took ",elapsed_time, " Seconds")
		print("The y value for the entered x is: ", y)
		plt.scatter(x, y, s=200, color='green')
		plt.scatter(lx, ly, color='blue')
		plt.plot(xe, ye, color='royalblue')
		plt.title('Lagrange Method', fontsize=18)
    	plt.xlabel(axisx, fontsize=16)
    	plt.ylabel(axisy, fontsize=16)
		# plt.scatter(x, y, s=200, color='green')
		plt.show()
	
	main_lg()

main()

# Tkinter GUI
# def onClick():
# 	#print(lr_chbx.get())
    
# 	dataset_entered = dataset_entry.get() + ".csv"
# 	axisx_entered = axisx_entry.get()
# 	axisy_entered = axisy_entry.get()
# 	type_entered = type_entry.get()
	
# 	if type_entered == "p":
# 		try:
# 			polynomial_regression(dataset_entered,axisx_entered, axisy_entered, 4)
# 		except:
# 			"There's an error somewhere!"
# 	else:
# 		try:
# 			linear_regression(dataset_entered,axisx_entered, axisy_entered)
# 		except:
# 			"error"	



# window = Tk()
# window.title("MLNA")
# window.configure(background = "#2b2b2b")

# type_txt = Label(window,width=30, bg="gray")
# type_txt.grid(row=1, column=0, sticky=W)

# type_entry = Entry(window, width=30, bg="white")
# type_entry.grid(row=1, column=0,sticky=W)

# dataset_txt = Label(window, text="Enter a file name:", font="none 12 bold")
# dataset_txt.grid(row=2, column=0,sticky=W)

# dataset_entry = Entry(window, width=30, bg="white")
# dataset_entry.grid(row=3, column=0,sticky=W)

# axisx_txt = Label(window, text = "Enter a column name from the dataset:", font = "none 12 bold")
# axisx_txt.grid(row=4, column=0, sticky=W)

# axisx_entry = Entry(window, width=30)
# axisx_entry.grid(row=5, column= 0, sticky=W)

# axisy_txt = Label(window, text = "Enter another column from the dataset:", font = "none 12 bold")
# axisy_txt.grid(row=4, column=3, sticky=W)

# axisy_entry = Entry(window, width=30)
# axisy_entry.grid(row=5, column= 3, sticky=W)

# submit_btn = Button(window, text="SUBMIT", width=6, command=onClick)
# submit_btn.grid(row=10, column=2,sticky=W)

# window.mainloop()

