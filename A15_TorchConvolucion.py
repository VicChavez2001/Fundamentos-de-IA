#===========================================
#   Ejemplo de red neuronal convolucinal
#===========================================
#   Traducido de pytorch tutorial 2023
#===========================================
#   Chavez Torres Victor Alexandro
#   Fundamentos de IA
#   ESFM IPN Mayo 2025
#===========================================
import torch 
import torch.nn as nn 
import torch.nn.functinonal as F
import torchvision
import torchivision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
#===========================================

#===========================
#   Configuracion del CPU
#===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#=====================
#   Hiper-parametros
#=====================
num_epochs = 10             # Iteraciones obre los datos de entrenamiento
batch_size = 4              # Subconjuntos de datos
learning_rate = 0.001       # Tasa de aprendizaje

#=================================================================
#   Definir pre-procesamiento de datos (transformacion)
#=================================================================
#   La base de datos tiene imagenes PILimage en el rango [0, 1]
#   Las transformamos a Tensores de rango normlaizado [-1, 1]
#=================================================================
transform = transforms.Compose(
    [transforms.ToTensor(), 
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#======================================================================================
#   CIFAR10: 60000 32x32 imagenes a color en c10 clases, con 6000 imagenes por clase
#======================================================================================
train_dataset = torchvision.datasets.CIFAR10(root='/.data', train=True,
                                             download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='/.data', train=False,
                                             download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=True)

test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=False)

#===========================
#   Objetos a clasificar
#===========================
classes = ('plane', ' car', ' bird', 'cat', 
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#=============================
#   Graficar con matplotlib
#=============================
def imshow(img): 
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
#===========================================
#   Obtener algunas imagenes para entrenar
#===========================================
dataiter = iter(train_loader)
images, labels = next(dataiter)

#===================================
#   Mostrar contenido en imagenes
#===================================
imshow(torchvision.utils.make_grid(images))

#===============================
#   Red neuronal convolucional
#===============================
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet(), self).__init__()
        # 3 entradas (a color), 6 salidas (filtros), 5x5 entradas en el kernel de convolucion
        self.convl = nn.Conv2d(3, 6, 5)
        # Maximo de una ventana de 2x2
        self.pool = nn.MaxPoll2d(2,2)
        # 6 entradas, 16 salidas (filtros), 5x5 entradas en el kernel de convolucion
        self.conv2 = nn.Conv2d(6, 16, 5)
        # redes lineales (entradas, salidas)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x): 
        # -> n, 3, 32, 32
        # maxpool( relu ( convolucion ) )
        x = self.pool(F.relu(self.conv1(x)))    # -> n, 6, 14, 14 
        x = self.pool(F.relu(self.conv2(x)))    # -> n, 16, 5, 5
        # reorganizar el tensor x 
        x = x.view(-1, 16 * 5 * 5)              # -> n, 400
        # redes lineales + relu 
        x = F.relu(self.fc1(x))                 # -> n, 120
        x = F.relu(self.fc2(x))                 # -> n, 84
        # red lineal
        x = self.fc3(x)                         # -> n, 10
        return x
    
#===============================
#   Correr el modelo enel GPU
#===============================
model = ConvNet().to(device)

#===========================================================================
#   Usar cross-entropy como costo y gradiente estocastico como optimizador
#===========================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#================================
#   Iteraciones (entrenamiento)
#================================
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # formato original: [4, 3, 32, 32] = 4, 3, 1024
        # capa de entrada: 3 canales de entrada, 6 canales de salida, 5 tamaño del kernel
        
        # imagenes
        images = images.to(device)
        
        # etiquetas
        labels = labels.to(device)
        
        # Evaluacion (forward pass)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Gradiente y optimizacion (backward)
        # inicializar gradiente a cero + calcularlo + aplicarlo
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 == 0: 
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            
print('Entrenamiento completo')

#==============================================
#   Guardar resultado del modelo (parametros)
#==============================================
PATH = './cnn.pth'
torch.save(model.state_dict(), PATH)

#=====================
#   Probar el modelo
#=====================
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for iamges, labels in test_loader: 
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max regresa (valor, inidice)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size): 
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 0
                n_class_samples[label] += 1
    
    acc = 100.0 * n_correct / n_samples
    print(f'Precision del modelo: {acc} %')
    
    for i in range(10): 
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Precision de {classes[i]}: {acc} %')
    