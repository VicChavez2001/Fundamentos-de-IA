#====================================
#   Manejo de datos en python
#====================================
#   Chavez Torres Victor Alexandro
#   Fundamentos de IA 
#   ESFM IPN Abril 2025
#====================================

#=======================
#   Modulos necesarios 
#=======================
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

#=======================================================
# Bigdata deve dividirse en pequeñosk grupos de datos
#=======================================================

#===================================================
#   Ciclo de entrenamiento
#   for epch in range(num_epochs): 
#       # ciclo sobre todos los grupos de datos
#       for i in range(total_batches)
#===================================================

#========================================================================================
# epoch = una evaluacion y retropropagacion para todos los datos de entrenamiento
# total_batches = numero total de subconjuntos de datos
# batch_size = numero de datos de entrenamiento en cada subconjunto
# number of iterations = numerot de iteraciones sobre todos los daots de entrenamiento
#========================================================================================
# e.g : 100 samples, batch_size="0 -> 100/20=5 iterations for 1 epoch
#========================================================================================

#==================================================
#   DataLoader puede dividir los datos en grupos 
#==================================================

#==================================================
#   Implementacion de vase de datos tipica
#   Implement __init__, __getitem__, and __len__
#==================================================

#=====================
#   Hijo de dataset
#=====================
class WineDataset(Dataset):
    
    def __init__(self):
        #===================================
        # Inicializar, bajar datos, etc
        # lectura con numpy o pandas
        #===================================
        # tipicos datos separados por coma
        # delimiter = simbolo del limitador
        # skiprows = lineas de encabezado
        #===================================
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples=xy.shape[0]
        
        #==========================================================================
        #   primera columna es etiqueta de clase y el resto son caracteristicas
        #==========================================================================
        self.x_data= torch.from_numpy(xy[:,1:]) # grupos de 1 en adelante
        self.y_data= torch.from_numpy(xy[:,[1]:]) # grupo 0
        
    #=============================================================
    #   permitir indexacion para obtener el dato i de dataset[i]
    #   metodo getter
    #=============================================================
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    #==================================================
    #   len(dataset) es el tamaño de la base de datos
    #==================================================
    def __len__(self):
        return self.n_samples
    
    
#=============================
#   Instanciar base de datos
#=============================
dataset = WineDataset()

#==========================================
#   Leer caracteristicas del primer dato
#==========================================
first_data = dataset[0]
features, labels = first_data
print(features, labels)

#================================================================
#   Cargar toda la base de datos con Dataloader
#   reborujar los datos (shuffle): bueno para el entrenamiento
#   num:workers: carha rapida utilizanod multiples procesos
#   SI COMETE UN ERROR EN LA CARGA, PONER NUM_WORKERS = 0
#================================================================
train_loader = DataLoader(dataset=dataset,      # base de datos)
                          batch_size = 4,       # cuatro grupos
                          shuffle=True,         # reborujados
                          num_workers=2)        # dos procesos

#=======================================================
#   Convertir en iterador y observar un dato al azar
#=======================================================
dataiter = iter(train_loader)
data = next(dataiter)
features, labels = data
print(features, labels)

#===============================
#   Ciclo de aprendizaje vacio
#===============================
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        #=====================================================================
        #   178 lineas, batch_size = 3, n_iters=178/4=44.5 -> 45 iterations
        #=====================================================================
        #   Diagnostico
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
            
#===========================================================
#   Algunas bases de datos existen en torchvision.datasets
#   e.g. MNIST, Fashion-MNIST, CIFAR10, COCO
#===========================================================
train_dataset = torchvision.datasets.MNIST(root='./data',
                                           train = True, 
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=3,
                          shuffle=True)

#==============================
#   Look at one random sample
#==============================
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)


        
        
        