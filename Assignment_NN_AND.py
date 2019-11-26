#Red neuronal 
#Compuerta lógica AND
#Atoany Fierro

import numpy as np

# X = (entradas binarias), y = salida AND binaria
#Crear una base de datos, en donde sea un par de entradas y una salida para
#una operación lógica AND
#------------------|
#  x1 AND x2 = y   |
#------------------|
#  0      0    0   |
#  0      1    0   |
#  1      0    0   |
#  1      1    1   |
#------------------|
#INSERTA TU CÓDIGO AQUI---------------------------------------------------------
X = np.array(([None, None], [None, None], [None, None], [None, None]), dtype=float) # entradas
y = np.array(([None], [None], [None], [None]), dtype=float) # salidas deseadas
#INSERTA TU CÓDIGO AQUI---------------------------------------------------------


class Neural_Network(object):
  def __init__(self):
  #arquitectura
  #Realiza la siguiente arquiectura:
  #2 neuronas en la capa de entrada
  #3 neuronas en la capa oculta
  #1 neurona en la capa de salida
  #INSERTA TU CÓDIGO AQUI-------------------------------------------------------
    self.inputSize = None
    self.hiddenSize = None
    self.outputSize = None
  #INSERTA TU CÓDIGO AQUI-------------------------------------------------------

  #inicialización de los pesos
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # matriz de pesos de la primer capa oculta
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # matriz de pesos de la capa de salida

  def forward(self, X):
    #cálculo de las activaciones de las neuronas
    #La respuesta lineal de la primer capa oculta se calcula realizando el producto punto
    #de las entradas X con los pesos W1
    #La respuesta lineal de la capa de salida se calcula realizando el producto punto
    #de las entradas (a1) con los pesos W2
    #Calcula las respuestas lineales y sus respuestas no lineales utilizando la función sigmoide
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------
    #ayuda: producto punto -> np.dot(a, b)
    #       sigmoide -> self.sigmoid(a)
    self.z1 = None # respuesta lineal de las neuronas
    self.a1 = None # función de activación
    self.z2 = None # respuesta lineal de las neuronas
    a2 = None # función de activación
    return a2
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------

  def sigmoid(self, z):
    # función de activación sigmoide
    #Programa la función de activación sigmoide
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------
    sigmoide = None
    return sigmoide
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------


  def sigmoidPrime(self, z):
    #derivada de la función sigmoide
    #Programa la derivada de la función sigmoide
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------
    sigmoide_prima = None
    return sigmoide_prima
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------

  def backward(self, X, y, a2):
    # propagación hacia atras
    #Programa la derivada de a función de costo
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------
    self.a2_error = None # derivada de la función de costo
    self.a2_delta = self.a2_error*self.sigmoidPrime(a2) # aplicacion de la derivada de sigmoide
    #INSERTA TU CÓDIGO AQUI-------------------------------------------------------

    self.a1_error = self.a2_delta.dot(self.W2.T) # error de z2: que tanto cambia mi función de costo si los parámetros en z2 cambia
    self.a1_delta = self.a1_error*self.sigmoidPrime(self.a1) # aplicando la derivada de sigmoide al error de z2

    self.W1 += X.T.dot(self.a1_delta) # modificando los parámetros de la primera capa oculta
    self.W2 += self.a1.T.dot(self.a2_delta) # modificando los parámetros de la capa de salida

  def train(self, X, y):
    #entrenamiento de la red
    a2 = self.forward(X)
    self.backward(X, y, a2)

  def saveWeights(self):
    # los pesos se guardan
    np.savetxt("w1.txt", self.W1, fmt="%s")
    np.savetxt("w2.txt", self.W2, fmt="%s")

  def predict(self):
    # predicciones
    print ("Predicción después de entrenamiento: ");
    print ("Entrada: \n" + str(X));
    print ("Salida real: \n" + str(self.forward(X)));

#se crea una instancia de la clase Neural_Network
NN = Neural_Network()
#Mofidica el número de iteraciones hasta que observes que la red ha converdigo
#INSERTA TU CÓDIGO AQUI-------------------------------------------------------
for i in range(10): # número de interaciones
#INSERTA TU CÓDIGO AQUI-------------------------------------------------------
  print ("Iteración: " + str(i))
  print ("Entrada: \n" + str(X))
  print ("Salida deseada: \n" + str(y))
  print ("Salida real: \n" + str(NN.forward(X)))
  print ("Error: " + str(np.mean(np.square(y - NN.forward(X))))) # función de costo mean sum squared
  print ()
  NN.train(X, y)

NN.saveWeights()
NN.predict()



