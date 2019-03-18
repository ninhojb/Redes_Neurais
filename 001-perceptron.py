'''
         Algoritmo
Enquanto o erro for diferente de zero
   Para cada registro
     Calcula a saída com os pesos atuais
     Compara a saída esperada com a saída calculada, somando o erro
         Para cada peso da rede
            Atualiza o peso - peso(n + 1) = peso(n) + (taxaAprendizagem * entrada * erro)

'''
import numpy as np

class Percepton:

    def __init__(self, entradas, saidas, pesos, taxa_aprendizagem=0.1):

        self.entradas = entradas
        self.saidas = saidas
        self.pesos = pesos
        self.taxa_aprendizagem = taxa_aprendizagem

    def step_Function(self,registro):
        return 1 if registro >=1 else 0

    def calculaSaida(self,s):
        soma = s.dot(self.pesos)
        return self.step_Function(soma)

    def treinar(self):
        erroTotal = 1

        while erroTotal != 0:

            erroTotal = 0
            for i in range(len(self.saidas)):
                resultado_saida = self.calculaSaida(np.asarray(self.entradas[i]))
                erro = abs(self.saidas[i] - resultado_saida)
                erroTotal+= erro
                for j in range(len(self.pesos)):
                    self.pesos[j] = self.pesos[j] + (
                        self.taxa_aprendizagem * self.entradas[i][j] * erro
                    )
                print('Peso Atual: ' + str(self.pesos[j]))
            print('Total de erros: ' + str(erroTotal))

'''
Rede Perceptron de uma camada
usando o operador E
'''
entrada = np.array([[0,0],[0,1],[1,0],[1,1]])
saida = np.array([0,0,0,1])
pesos = np.array([0.0,0.0])

teste = Percepton(entrada, saida,pesos)

teste.treinar()
'''
após rede treinada, entra com os valores
'''
print('\nRede neural Treinada')
print(teste.calculaSaida(entrada[0]))
print(teste.calculaSaida(entrada[1]))
print(teste.calculaSaida(entrada[2]))
print(teste.calculaSaida(entrada[3]))



