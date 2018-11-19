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
    lx = list(map(float, input("Lista de valores de x: ").split()))
    ly = list(map(float, input("Lista de valores de y: ").split()))
    y = lagrange(x, lx, ly)
    print("El resultado es: ", y)
    input("Presione una tecla para finalizar")

main()
