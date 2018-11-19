import matplotlib.pyplot as plt
import pandas as pd

def lagrange(x, lx, ly):
    y = 0
    for i in range(len(lx)):
        L = 1
        for j in range(len(lx)):
            if  j != i:
                L = L*(x-lx[j])/(lx[i]-lx[j])
        y = y + L*ly[i]
    return y


def main():
    print("METODO DE LAGRANGE")
    x = float(input("Valor a interpolar: "))
    data = pd.read_csv("Temperature.csv", header=0)
    lx = data["Year"]
    ly = data["Temperature"]

    #lx = list(map(float, input("Lista de valores de x: ").split()))
    #ly = list(map(float, input("Lista de valores de y: ").split()))
    y = lagrange(x, lx, ly)
    print("El resultado es: ", y)

    plt.scatter(lx, ly, color='blue')
    plt.plot(lx, ly, color='red')
    plt.show()
    input("Presione una tecla para finalizar")
    
main()
