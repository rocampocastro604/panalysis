#NOT WORKING
def same_sign(a, b):
    return a*b > 0

def incremental_searches(f, value, delta): 
    xnew = value + delta
    #assert not same_sign(f(value), f(xnew))

    for i in range(54):
        if same_sign:
            value = xnew
            xnew = value + delta
        else:
            print("["+ value , xnew + "]")
        return xnew 
   
def f(x):
    return -26 + 85*x -95 * x**2 + 44*x**3 + x**5
    #return function    

def main():    
    x = incremental_searches(f, 0.4, 0.1)
    print(x, f(x))
    
main()