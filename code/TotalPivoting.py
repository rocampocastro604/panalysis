from math import fabs
from math import sqrt
from sys import exit


def exchangeRows(M, r1, r2):

	M[r1], M[r2] = M[r2], M[r1]
	return M

def exchangeCols(M, c1, c2):

	for k in range(len(M)):
		M[k][c1] , M[k][c2] = M[k][c2], M[k][c1]
	return M

def pivot(a, P, Q, colum, piv=0):
	"""Pivoting in any mode"""

	if piv > 1 or piv < 0:
		exit('Invalid values for the pivoting method')
	n = len(a)

	temp = a[colum][colum]
	if piv == 0:
		row_maxColumn = maxColum(a, colum)
		if row_maxColumn != colum:
			a = pivotP(a, row_maxColumn, colum)
			P = exchangeRows(P, row_maxColumn, colum)
			print
			'P(%d,%d)' % (row_maxColumn, colum)
			printMatrix(Pr(n, row_maxColumn, colum))

	elif piv == 1:
		row, c = maxSubMatrix(a, colum)
		if (row != colum) or (c != colum):
			a = pivotT(a, colum)
			P = exchangeCols(P, row, colum)
			Q = exchangeCols(Q, colum, c)
			print
			'P(%d,%d):' % (colum, row)
			printMatrix(Pr(n, row, colum))
			print
			'Q(%d,%d):' % (colum, c)
			printMatrix(Pr(n, c, colum))

	# return the matrix with the respected order
	return a, P, Q
def pivotP(M, r1, r2):
	"""It changes the rows to find the maximun value"""

	return exchangeRows(M, r1, r2)

def pivotT(M, i):
	"""It searches the maximun element of submatrix A[i] and changes columns and rows"""

	r, c = maxSubMatrix(M, i)
	M = pivotP(M, i, r)

	return exchangeCols(M, c, i)

def Pr(n, r1, r2):

	#Identity matrix
	I = [[float(i == j) for j in range(n)] for i in range(n)]
	return exchangeRows(I, r1, r2)