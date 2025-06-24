import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time

# Función auxiliar para definir ticks en múltiplos de π en el eje x
def set_pi_ticks(ax, xmax):
    ticks = np.arange(0, xmax + np.pi, np.pi)
    labels = [ "0" if i==0 else rf"${i}\pi$" for i in range(len(ticks)) ]  # Añadida 'r'
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

# Definición de la arquitectura de la red (se usará para ambos modelos)
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)
        self.activation = nn.Tanh()
        self._init_weights()  

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.activation(self.fc5(out))
        out = self.fc6(out)
        return out

    def compute_derivatives(self, x):
        # Se requiere que x tenga gradiente para calcular derivadas
        x = x.clone().detach().requires_grad_(True)
        y = self(x)
        # Primera derivada
        dy_dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        # Segunda derivada
        d2y_dx2 = torch.autograd.grad(dy_dx, x, grad_outputs=torch.ones_like(dy_dx), create_graph=True)[0]
        return y, dy_dx, d2y_dx2

# Fijamos semillas para reproducibilidad
np.random.seed(42)
torch.manual_seed(42)

epochs = 3000

# Definición de puntos para PDE
N_pde = 50
x_pde_orig = np.linspace(0, 4*np.pi, N_pde).reshape(-1, 1)
x_pde_t = torch.tensor(x_pde_orig, dtype=torch.float32)

# Peso para las pérdidas
w_pde = 0.5   # para la PDE
w_bc  = 0.5   # para las condiciones de contorno

# -------------------------------
# 1. Entrenamiento del modelo PINN (PDE + BC)
# -------------------------------
print("Iniciando entrenamiento PINN (PDE + BC)...")
start_time_PINN = time.time()

model_PINN = PINN()
optimizer_PINN = optim.Adam(model_PINN.parameters(), lr=0.001)

loss_train_list = []
loss_pde_list   = []
loss_bc_list    = []

for epoch in range(epochs):
    model_PINN.train()
    optimizer_PINN.zero_grad()
    
    # PDE (y'' + y = 0)
    y_pde, dy_dx, d2y_dx2 = model_PINN.compute_derivatives(x_pde_t)
    loss_pde = torch.mean((d2y_dx2 + y_pde)**2)
    
    # Condiciones de contorno en x = 0:
    x_bc = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    y_bc, dy_dx_bc, _ = model_PINN.compute_derivatives(x_bc)
    loss_bc = (y_bc - 1)**2 + (dy_dx_bc - 0)**2
    loss_bc = torch.mean(loss_bc)
    
    # Pérdida total
    loss = w_pde * loss_pde + w_bc * loss_bc
    loss.backward()
    optimizer_PINN.step()
    
    loss_train_list.append(loss.item())
    loss_pde_list.append(loss_pde.item())
    loss_bc_list.append(loss_bc.item())
    
    if epoch % 500 == 0:
        print(f"PINN - Epoch {epoch}: Loss Total = {loss.item():.6f} | PDE Loss = {loss_pde.item():.6f}, BC Loss = {loss_bc.item():.6f}")

end_time_PINN = time.time()
training_time_PINN = end_time_PINN - start_time_PINN
print(f"Tiempo de entrenamiento PINN: {training_time_PINN:.2f} segundos")

# -------------------------------
# 2. Entrenamiento del modelo NN simple (Solo BC)
# -------------------------------
print("\nIniciando entrenamiento NN simple (Solo BC)...")
start_time_NN = time.time()

model_NN = PINN()  # Se usa la misma arquitectura
optimizer_NN = optim.Adam(model_NN.parameters(), lr=0.001)

loss_NN_list = []

for epoch in range(epochs):
    model_NN.train()
    optimizer_NN.zero_grad()
    
    # Solo condiciones de contorno en x = 0
    x_bc = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    y_bc, dy_dx_bc, _ = model_NN.compute_derivatives(x_bc)
    loss_bc = (y_bc - 1)**2 + (dy_dx_bc - 0)**2
    loss_bc = torch.mean(loss_bc)
    
    loss_NN = w_bc * loss_bc
    loss_NN.backward()
    optimizer_NN.step()
    
    loss_NN_list.append(loss_NN.item())
    
    if epoch % 500 == 0:
        print(f"NN Simple - Epoch {epoch}: BC Loss = {loss_bc.item():.6f}")

end_time_NN = time.time()
training_time_NN = end_time_NN - start_time_NN
print(f"Tiempo de entrenamiento NN: {training_time_NN:.2f} segundos")

# -------------------------------
# 3. Gráficas finales
# -------------------------------

# Definir malla fina para evaluar 
x_fine = np.linspace(0, 6*np.pi, 200).reshape(-1, 1)
x_fine_t = torch.tensor(x_fine, dtype=torch.float32)

with torch.no_grad():
    # Salida del modelo PINN (PDE + BC)
    y_PINN = model_PINN(x_fine_t).detach().numpy()
    # Salida del modelo NN simple (Solo BC)
    y_NN = model_NN(x_fine_t).detach().numpy()

y_exacta = np.cos(x_fine)

# Gráfica 1: Evolución de pérdidas PINN (escala logarítmica)
plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.semilogy(loss_train_list, 'k-', label='Loss Total PINN', linewidth=2)
plt.semilogy(loss_pde_list, 'g-', label='L_PDE PINN', linewidth=1.5)
plt.semilogy(loss_bc_list, 'b-', label='L_BC PINN', linewidth=1.5)
plt.xlabel('Épocas')
plt.ylabel('Loss (escala log)')
plt.title('Evolución Pérdidas PINN')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 2: Evolución de pérdidas NN (escala logarítmica)
plt.subplot(1, 3, 2)
plt.semilogy(loss_NN_list, 'r-', label='Loss BC NN', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Loss (escala log)')
plt.title('Evolución Pérdidas NN Simple')
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfica 3: Comparación pérdidas totales PINN vs NN
plt.subplot(1, 3, 3)
plt.semilogy(loss_train_list, 'b-', label='Loss Total PINN', linewidth=2)
plt.semilogy(loss_NN_list, 'r-', label='Loss BC NN', linewidth=2)
plt.xlabel('Épocas')
plt.ylabel('Loss (escala log)')
plt.title('Comparación PINN vs NN')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('BC_comparacion_losses.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfica de comparación de funciones con puntos PDE y condiciones de contorno
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_fine, y_PINN, 'b-', label='PINN (PDE + BC)', linewidth=2)
ax.plot(x_fine, y_NN, 'm-', label='NN Simple (Solo BC)', linewidth=2)
ax.plot(x_fine, y_exacta, 'k--', label='Solución exacta: cos(x)', linewidth=2)

# Puntos PDE en y=0 (puedes dejarlo si quieres visualizar la física)
ax.scatter(x_pde_orig, np.zeros_like(x_pde_orig), color='green', marker='o', 
           s=30, label='Puntos PDE', zorder=5, alpha=0.6)

# Punto de condición de contorno SOBRE la función (en x=0, y=1)
ax.scatter([0], [1], color='red', marker='D', s=100, 
           label='Condición: y(0)=1', zorder=6, edgecolors='darkred')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Comparación de modelos: PINN vs NN Simple (Solo BC)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
set_pi_ticks(ax, 6*np.pi)
plt.savefig('BC_comparacion_funciones.png', dpi=300, bbox_inches='tight')
plt.close()

# Gráfica de y_pred vs y_true para puntos específicos
# Evaluamos en algunos puntos conocidos
x_test_points = np.array([[0], [np.pi/2], [np.pi], [3*np.pi/2], [2*np.pi]]).reshape(-1, 1)
y_test_exact = np.cos(x_test_points)
x_test_t = torch.tensor(x_test_points, dtype=torch.float32)

with torch.no_grad():
    y_test_pred_PINN = model_PINN(x_test_t).detach().numpy()
    y_test_pred_NN = model_NN(x_test_t).detach().numpy()

plt.figure(figsize=(8, 6))
plt.scatter(y_test_exact, y_test_pred_PINN, c='blue', label='PINN (PDE+BC)', alpha=0.7, s=80)
plt.scatter(y_test_exact, y_test_pred_NN, c='red', label='NN (Solo BC)', alpha=0.7, s=80)
plt.plot([y_test_exact.min(), y_test_exact.max()],
         [y_test_exact.min(), y_test_exact.max()], 'k--', alpha=0.8)
plt.xlabel('y_true')
plt.ylabel('y_pred')
plt.title('y_pred vs y_true (Puntos de test)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('BC_ypred_vs_ytrue.png', dpi=300, bbox_inches='tight')
plt.close()

# Graficar residuo PDE vs x en malla fina [0,6π]
x_residuo = np.linspace(0, 6*np.pi, 300).reshape(-1, 1)
x_residuo_t = torch.tensor(x_residuo, dtype=torch.float32)
y_res, dy_dx_res, d2y_dx2_res = model_PINN.compute_derivatives(x_residuo_t)
residuo = d2y_dx2_res + y_res  
residuo = residuo.detach().numpy()

plt.figure(figsize=(10,6))
plt.plot(x_residuo, residuo, 'r-', label='Residuo PDE', linewidth=2)
plt.xlabel('x')
plt.ylabel('Residuo')
plt.title('Residuo de la PDE vs x (modelo PINN)')
plt.legend()
plt.grid(True, alpha=0.3)
set_pi_ticks(plt.gca(), 6*np.pi)
plt.savefig('BC_residuo_PDE.png', dpi=300, bbox_inches='tight')
plt.close()

# Cálculo de métricas finales para comparar
# Evaluación en intervalos para información adicional
x_train_interval = np.linspace(0, 2*np.pi, 100).reshape(-1, 1)
x_extrapolation = np.linspace(2*np.pi, 4*np.pi, 100).reshape(-1, 1)
x_global = np.linspace(0, 4*np.pi, 200).reshape(-1, 1)
        
x_train_interval_t = torch.tensor(x_train_interval, dtype=torch.float32)
x_extrapolation_t = torch.tensor(x_extrapolation, dtype=torch.float32)
x_global_t = torch.tensor(x_global, dtype=torch.float32)
        
with torch.no_grad():
    # Predicciones PINN
    y_train_interval_pred_PINN = model_PINN(x_train_interval_t).detach().numpy()
    y_extrapolation_pred_PINN = model_PINN(x_extrapolation_t).detach().numpy()
    y_global_pred_PINN = model_PINN(x_global_t).detach().numpy()
    
    # Predicciones NN
    y_train_interval_pred_NN = model_NN(x_train_interval_t).detach().numpy()
    y_extrapolation_pred_NN = model_NN(x_extrapolation_t).detach().numpy()
    y_global_pred_NN = model_NN(x_global_t).detach().numpy()
        
y_train_interval_exact = np.cos(x_train_interval)
y_extrapolation_exact = np.cos(x_extrapolation)
y_global_exact = np.cos(x_global)

# MSE para PINN        
mse_train_interval_PINN = mean_squared_error(y_train_interval_exact, y_train_interval_pred_PINN)
mse_extrapolation_PINN = mean_squared_error(y_extrapolation_exact, y_extrapolation_pred_PINN)
mse_global_PINN = mean_squared_error(y_global_exact, y_global_pred_PINN)

# MSE para NN
mse_train_interval_NN = mean_squared_error(y_train_interval_exact, y_train_interval_pred_NN)
mse_extrapolation_NN = mean_squared_error(y_extrapolation_exact, y_extrapolation_pred_NN)
mse_global_NN = mean_squared_error(y_global_exact, y_global_pred_NN)

# MSE en puntos de test específicos
mse_test_PINN = mean_squared_error(y_test_exact, y_test_pred_PINN)
mse_test_NN = mean_squared_error(y_test_exact, y_test_pred_NN)

# Resultados finales
resultados = {
    'MSE Train Interval (PINN)': mse_train_interval_PINN,
    'MSE Extrapolación (PINN)': mse_extrapolation_PINN,
    'MSE Global (PINN)': mse_global_PINN,
    'MSE Test Points (PINN)': mse_test_PINN,
    'MSE Train Interval (NN)': mse_train_interval_NN,
    'MSE Extrapolación (NN)': mse_extrapolation_NN,
    'MSE Global (NN)': mse_global_NN,
    'MSE Test Points (NN)': mse_test_NN,
    'Tiempo Entrenamiento PINN (s)': training_time_PINN,
    'Tiempo Entrenamiento NN (s)': training_time_NN,
}

print(f"\n=== RESULTADOS FINALES (Solo BC) ===")
print(f"PINN - Total Train Loss Final: {loss_train_list[-1]:.6f}")
print(f"NN - BC Loss Final: {loss_NN_list[-1]:.6f}")
print(f"\n--- COMPARACIÓN MSE INTERVALOS ---")
print(f"Intervalo [0,2π] - PINN: {mse_train_interval_PINN:.5f} | NN: {mse_train_interval_NN:.5f}")
print(f"Extrapolación [2π,4π] - PINN: {mse_extrapolation_PINN:.5f} | NN: {mse_extrapolation_NN:.5f}")
print(f"Global [0,4π] - PINN: {mse_global_PINN:.5f} | NN: {mse_global_NN:.5f}")
print(f"Puntos Test - PINN: {mse_test_PINN:.5f} | NN: {mse_test_NN:.5f}")
print(f"\n--- MEJORA RELATIVA PINN vs NN ---")
print(f"Intervalo: {((mse_train_interval_NN - mse_train_interval_PINN)/mse_train_interval_NN)*100:.1f}% mejor PINN")
print(f"Extrapolación: {((mse_extrapolation_NN - mse_extrapolation_PINN)/mse_extrapolation_NN)*100:.1f}% mejor PINN")
print(f"Global: {((mse_global_NN - mse_global_PINN)/mse_global_NN)*100:.1f}% mejor PINN")
print(f"Test Points: {((mse_test_NN - mse_test_PINN)/mse_test_NN)*100:.1f}% mejor PINN")
print(f"\n--- TIEMPOS ---")
print(f"Tiempo entrenamiento PINN: {training_time_PINN:.2f}s")
print(f"Tiempo entrenamiento NN: {training_time_NN:.2f}s")
print(f"Ratio tiempo PINN/NN: {training_time_PINN/training_time_NN:.2f}x")

print("\nGráficas guardadas con prefijo 'BC_'.")