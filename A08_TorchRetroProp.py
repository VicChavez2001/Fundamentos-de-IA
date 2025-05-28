#==================================
#   Gradientes simples con pytorch
#==================================
#   Chavez Torres Victor Alexandro
#   Fundamentos de IA
#   ESFM IPN Abril 2025
#==================================
import torch 

x = torch.tensor(1.0)
y = torch.tensor(2.0)

#=============================================================
#   Variable de diferenciacion (d/dw) -> requires_grad=True
#=============================================================
w = torch.tensor(1.0, requires_grad=True)

#==================================
#   Evaluacion calculo de costo
#==================================
y_predicted = w * x
loss = (y_predicted - y)**2
print(loss)

#=============================================
#   Retropropagacion para calcular gradiente
#   w.grad es el gradiente
#=============================================
loss.backward()
print(w.grad)

#================================================
#   Nuevos coeficientes (descenso de gradiente)
#   Reoetir evaluacion y retropropagacion
#================================================
with torch.no_grad():
    w -= 0.01 * w.grad
w.grad.zero_()
print(w)