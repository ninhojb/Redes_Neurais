from sklearn.neural_network import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
entradas = iris.data
saidas = iris.target


redeNeural = MLPClassifier(
    verbose=True, max_iter=10000, tol=0.00001, activation='logistic'

)

redeNeural.fit(entradas,saidas)
print('\nRede Treinada')

teste = redeNeural.predict([[5,7,5.1,7.2]])

print(teste)
if teste == 0:
    print('Setosa')
elif teste == 1 :
    print('Versicolor')
else:
    print('Virginica')