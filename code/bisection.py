def same_sign(a, b):
	return a*b > 0


def bisection(func, low, high):
	
	#Returns an error if the current interval has no root
	assert not same_sign(func(low), func(high))

	for i in range(54):
		xm = (low + high)/2.0
		if same_sign:
			low = xm
		else:
			high = xm
	return xm

def f(x):
	#Here goes the function that we want
	return -26 + 85*x -95 * x**2 + 44*x**3 + x**5


def main():
	x = bisection(f, 0, 1)
	print(x, f(x))

main()