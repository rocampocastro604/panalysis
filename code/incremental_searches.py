def incremental_searches(): 
    function = input("Type a function to find its roots: ")
    xo = float(input("Type an initial value: "))
    delta = float(input("Type a DELTA value: "))
    it = 100
    yo = replace(x in function with xo)
    #yo = function(xo)

    if yo == 0:
        print(xo + " is a root")
    else:
        x1 = delta + xo
        contador = 1
        y1 = function(x1)

        while (yo*y1 > 0) & (contador<iter):
            xo = x1
            yo = y1
            x1 = xo + delta
            y1 = function(x1)
            contador = contador + 1
        
    if y1 == 0:
        print("Root in: " + x0)
    elif yo *y1 < 0:
        print("Its a root around: " + x0 + "and" + x1)
    else:
        print("MAX Iterations")
    
incremental_searches()