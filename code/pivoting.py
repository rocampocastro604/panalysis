import numpy
m = int(input('Valor de m: '))
n = int(input('Valor de n: '))
matrix = numpy.zeros((m,n))
vector = numpy.zeros((n))
x = numpy.zeros((m))
print('Introduce la matriz de coeficientes y el vector solución: ')
for r in range(0, m):
	for c in range(0, n):
		matrix[(r),(c)]=(input("Elemento a["+ str(r+1)+ ","+str(c+1)+"]: "))
	vector[(r)]=(input('b['+str(r+1) + ']: '))
print(matrix)

for k in range(0,m):
	for r in range(k+1,m):
		factor=(matrix[r,k]/matrix[k,k])
		vector[r]=vector[r]-(factor*vector[k])
		for c in range(0,n):
			matrix[r,c]=matrix[r,c]-(factor*matrix[k,c])

#Sustitución pa atrás
x[m-1]=vector[m-1]/matrix[m-1, m-1]
print(x[m-1])

for r in range(m-2,-1, -1):
	suma = 0
	for c in range(0,n):
		suma=suma+matrix[r,c]*x[c]
	x[r]=(vector[r]-suma)/matrix[r, r]

print ('Resultado matriz')
print(matrix)
print ('Resultado del vector')
print(vector)
print ('Resultados: ')
print(x)