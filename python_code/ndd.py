import matplotlib.pyplot as plt
import pandas as pd
def diferencias_divididas(x, lx, ly):
    """Metodo numerico de dfierencias dividadas 
    
    Arguments:
    - `x`: Valor a interpolar
    - `lx`: Lista con los valores de x
    - `ly`: Lista con los valores de y (f(x))
    """
    y = 0
    for i in range(len(lx)-1):
        if x >= lx[i] and x <= lx[i+1]:
            y = (ly[i+1] - ly[i]) / (lx[i+1]-lx[i]) * (x - lx[i]) + ly[i]
    return y

   
def main():
    print("METODO DE DIFERENCIAS DIVIDIDAS")
    x = float(input("Valor a interpolar: "))
    data = pd.read_csv("Temperature.csv", header=0)
    lx = data["Year"]
    ly = data["Temperature"]
    #lx = list(map(float, input("Lista de valores de x: ").split()))
    #ly = list(map(float, input("Lista de valores de y: ").split()))
    y = diferencias_divididas(x, lx, ly)
    print("El resultado es: ", y)
    
    plt.scatter(lx, ly, color='blue')
    plt.plot(lx, ly, color='red')
    plt.show()
    input("Presione una tecla para finalizar")
main()

