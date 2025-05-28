#===================================================================
#   INtroduccin al uso de softmax y cross entropy loss en pytorch
#===================================================================
#   Chavez Torres Victor Alexandro
#   Fundamentos de IA
#   ESFM IPN Mayo 2025
#===================================================================

#========================
#   Modulos necesarios
#========================
import torch 
import torch.nn as nn
import numpy as np

#===========================================================================
#   Modelo de Boltzman
#========================
#   En termodinamica es la probabilidad de encontrar un sistema en algun
#   estado dada su energia y temperatura
#===========================================================================
#           -> 2.0                  ->0.65
# Linear    -> 1.0      -> Softmax  ->0.25      ->CrossEntropy(y, y_hat)
#           -> 0.1                  ->0.1
#
#       puntajes(logits)        probabilidades
#                               suma = 1.0
#===========================================================================

#===========================================================================
#   Softmax aplica el modelo de distribucion exponencial para cada elemento
#   normalizada con la suma de todas las exponenciales
#===========================================================================
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

#=================
#   Vector en R3
#=================
x = np.array([2.0, 1.0, 0.1])

#====================================
#   softmax de elementos del vector
#====================================
outputs = softmax(x)
print('softmax numpy:', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # tomar softmax de los elementos del eje 0
print('softmax torch', outputs)

#======================================================================================
#   Cross-entropy loss, o log loss, mide el rendimiento de un modelo de clasificacion
#   cuya salida es un valor de probabilidad entre 0 y 1
#======================================================================================
#   Se incrementa conforme la probabilidad diverge del nivel veradero
#======================================================================================
def cross_entropy(actual, predicted): 
    EPS = 1e-15
    # limitar los valores a un minimo EPS y maximo 1-EPS
    predicted = np.clip(predicted, EPS, 1 - EPS)
    # calculo del rendimiento
    loss = -np.sum(actual * np.log(predicted))
    return loss # \ float(predicted.shape[0])

#=====================================
# y debe ser alguna de las opciones
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]
#=====================================
Y = np.array([1, 0, 0])
Y_pred_good = np.array([0.7, 0.2, 0.1])
Y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(Y, Y_pred_good)
l2 = cross_entropy(Y, Y_pred_bad)
print(f'Loss1 numpy: {l1:.4f}')
print(f'Loss2 numpy: {l2:.4f}')

#==================================================
#   CrossEntropyLoss en Pytorch (aplica softmax)
#   nn.LogSoftmax + nn.NLLLoss
#   NLLLoss = "negative log likehood loss"
#==================================================
loss = nn.CrossEntropyLoss()
#   loss(input, target)

#===========================================================================
#   objetivo es de tamaño nSamples = 1
#   cada elemento teine etiqueta de clase: 0, 1 o 2
#   Y (=objetivo) contiene etiquetas de clase class no opciones binarias
#===========================================================================
Y = torch.tensor([0])

#=====================================================================================
#   input es de tamaño nSamples x nClasses = 1 x 3
#   y_pred (=input) deben estar sin normalizar (logits) para cada clase, no softmax
#=====================================================================================
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])
#=======================================
#   usar loss = nn.CrossEntropyLoss()
#=======================================
l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(f'PyTorch Loss1: {l1.item():.4f}')
print(f'PyTorch Loss2: {l2.item():.4f}')

#=====================================================
#   Predicciones (regresa el maximo en la dimension)
#=====================================================
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')

#=======================================================================
#   permite calcular el rendimiento para multiples conjuntos de datos
#=======================================================================
#   vector objetivo es de tamaño nBatch = 3 
#   cada elemento tiene etiqueta de clase: 0, 1 or 2
#=======================================================================
Y = torch.tensor([2, 0, 1])


