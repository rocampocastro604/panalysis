

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
    lx = map(float, input("Lista de valores de x: ").split())
    ly = map(float, input("Lista de valores de y: ").split())
    y = diferencias_divididas(x, lx, ly)
    print("El resultado es: ", y)
    input("Presione una tecla para finalizar")

main()