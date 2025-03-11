#Clase 1: Regresion lineal
#==================================
#Victor Alexandro Chavez Torres
#Fundamentos de IA
#Matematica algoritmica
#ESFM IPN
#Febrero 2025
#=======================
#Modulos necesarios
#========================

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
#===============================

#Minimos cuadrados 
def minimos_cuadrados(X,Y):
    N = len(X)
    sumx = np.sum(X)
    sumy = np.sum(Y)
    sumxy = np.sum(X*Y)
    sumx2 = np.sum(X*X)
    w1 = (N * sumxy - sumx*sumy)/(N*sumx2 - sumx*sumx)
    w0 = (sumy - w1*sumx) / N
    Ybar = w0 + w1*X
    return Ybar, w0, w1
#======================
#Programa principal
#======================
if __name__ == "__main__":
    #====================
    #Leer datos
    #====================
    data = pd.read_csv('data.csv')
    X = data.iloc[:,0]
    Y = data.iloc[:,1]
    Ybar, w0, w1 = minimos_cuadrados(X,Y)
    #=====================
    #Grafica
    #=====================
    plt.scatter(X,Y)
    plt.rcParams['figure.figsize'] = (12.0, 9.0)
    plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()