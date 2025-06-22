import torch
import numpy as np
import time
from pinn import SchrodingerPINN
from scipy.special import hermite
import math
import matplotlib.pyplot as plt

L = 6.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n = int(input("Introduce el número de estado (n=0,1,2,...): "))

def psi_analitica(n, x):
    Hn = hermite(n)
    norm = 1.0 / np.sqrt((2.0**n) * math.factorial(n) * np.sqrt(np.pi))
    return norm * np.exp(-x**2 / 2) * Hn(x)

def eval_model(model_path, x_path, psi_path, label):
    # Cargar modelo
    model = SchrodingerPINN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    # Cargar puntos de entrenamiento usados
    x_train = np.load(x_path)
    puntos = len(x_train)
    # Evaluar función de onda en malla densa
    x_plot = torch.linspace(-L, L, 500, device=device).view(-1, 1)
    with torch.no_grad():
        psi_pred = model(x_plot).cpu().numpy().squeeze()
    norm_pred = np.sqrt(np.trapezoid(np.abs(psi_pred)**2, x_plot.cpu().numpy().squeeze()))
    psi_pred /= norm_pred
    # Analítica
    x_np = x_plot.cpu().numpy().squeeze()
    psi_ana = psi_analitica(n, x_np)
    mse = np.mean((psi_pred - psi_ana)**2)
    return mse, puntos, psi_pred, psi_ana, x_np

# Cargar tiempos automáticamente de los archivos de información
try:
    info_fija = np.load(f"training_info_n{n}_fija.npy", allow_pickle=True).item()
    tiempo_fija = info_fija['training_time']
    print(f"Tiempo malla fija cargado: {tiempo_fija:.2f} s")
except:
    print("Advertencia: No se pudo cargar el tiempo de malla fija, usando valor por defecto")
    tiempo_fija = 0.0

try:
    info_adap = np.load(f"training_info_n{n}_adaptativo.npy", allow_pickle=True).item()
    tiempo_adap = info_adap['training_time']
    print(f"Tiempo malla adaptativa cargado: {tiempo_adap:.2f} s")
except:
    print("Advertencia: No se pudo cargar el tiempo de malla adaptativa, usando valor por defecto")
    tiempo_adap = 0.0

# Evaluar malla fija
mse_fija, puntos_fija, psi_pred_fija, psi_ana, x_np = eval_model(
    model_path=f"modelo_n{n}_fija.pth",
    x_path=f"x_vals_n{n}_fija.npy",
    psi_path=f"psi_pred_n{n}_fija.npy",
    label="Fija"
)

# Evaluar malla adaptativa
mse_adap, puntos_adap, psi_pred_adap, _, _ = eval_model(
    model_path=f"modelo_n{n}_adaptativo.pth",
    x_path=f"x_vals_n{n}_adaptativo.npy",
    psi_path=f"psi_pred_n{n}_adaptativo.npy",
    label="Adaptativa"
)

# Cargar energia
energia_fija = np.load(f"energia_history_n{n}_fija.npy")
energia_adap = np.load(f"energia_history_n{n}_adaptativo.npy")
E_final_fija = energia_fija[-1]
E_final_adap = energia_adap[-1]
E_teorica = n + 0.5

# Calcular errores energéticos
error_fija = abs(E_final_fija - E_teorica) / E_teorica * 100
error_adap = abs(E_final_adap - E_teorica) / E_teorica * 100

print(f"- Energía final - Fija: {E_final_fija:.6f} (error: {error_fija:.2f}%)")
print(f"- Energía final - Adaptativa: {E_final_adap:.6f} (error: {error_adap:.2f}%)")

# Cargar historiales de loss y energía
loss_fija = np.load(f"loss_history_n{n}_fija.npy")
loss_adap = np.load(f"loss_history_n{n}_adaptativo.npy")
energia_fija = np.load(f"energia_history_n{n}_fija.npy")
energia_adap = np.load(f"energia_history_n{n}_adaptativo.npy")

# Mostrar resultados
print("Indep tiempo Ec schrodinger 1d:")
print(f"- Malla fija: Tiempo total de entrenamiento: {tiempo_fija:.2f} s, puntos: {puntos_fija}, MSE: {mse_fija:.2e}")
print(f"- Malla adaptativa: Tiempo total de entrenamiento: {tiempo_adap:.2f} s, puntos: {puntos_adap}, MSE: {mse_adap:.2e}")

# Calcular densidades de probabilidad
prob_fija = psi_pred_fija**2
prob_adap = psi_pred_adap**2
prob_ana = psi_ana**2

# Calcular MSE para densidades de probabilidad
mse_prob_fija = np.mean((prob_fija - prob_ana)**2)
mse_prob_adap = np.mean((prob_adap - prob_ana)**2)

print(f"- MSE función de onda - Fija: {mse_fija:.2e}, Adaptativa: {mse_adap:.2e}")
print(f"- MSE densidad probabilidad - Fija: {mse_prob_fija:.2e}, Adaptativa: {mse_prob_adap:.2e}")

# 1. Evolución de energía - Malla Fija
plt.figure(figsize=(10, 6))
plt.plot(energia_fija, label="Malla Fija", linewidth=2, color='blue')
plt.axhline(y=n+0.5, color='red', linestyle='--', label='Teórica')
plt.xlabel("Época")
plt.ylabel("Energía")
plt.title("Evolución Energía - Malla Fija")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"evolucion_energia_fija_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 2. Evolución de energía - Malla Adaptativa
plt.figure(figsize=(10, 6))
plt.plot(energia_adap, label="Malla Adaptativa", linewidth=2, color='orange')
plt.axhline(y=n+0.5, color='red', linestyle='--', label='Teórica')
plt.xlabel("Época")
plt.ylabel("Energía")
plt.title("Evolución Energía - Malla Adaptativa")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"evolucion_energia_adaptativa_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 3. Función de onda - Malla Fija vs Analítica
plt.figure(figsize=(10, 6))
plt.plot(x_np, psi_ana, 'k--', label="Analítica", linewidth=2)
plt.plot(x_np, psi_pred_fija, label="PINN Fija", linewidth=2, color='blue')
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title(f"Función de Onda - Malla Fija - Estado n={n}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"funcion_onda_fija_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 4. Función de onda - Malla Adaptativa vs Analítica
plt.figure(figsize=(10, 6))
plt.plot(x_np, psi_ana, 'k--', label="Analítica", linewidth=2)
plt.plot(x_np, psi_pred_adap, label="PINN Adaptativa", linewidth=2, color='orange')
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.title(f"Función de Onda - Malla Adaptativa - Estado n={n}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"funcion_onda_adaptativa_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 5. Densidad de probabilidad - Malla Fija vs Analítica
plt.figure(figsize=(10, 6))
plt.plot(x_np, prob_ana, 'k--', label="Analítica", linewidth=2)
plt.plot(x_np, prob_fija, label="PINN Fija", linewidth=2, color='blue')
plt.xlabel("x")
plt.ylabel("|ψ(x)|²")
plt.title(f"Densidad de Probabilidad - Malla Fija - Estado n={n}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"densidad_probabilidad_fija_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 6. Densidad de probabilidad - Malla Adaptativa vs Analítica
plt.figure(figsize=(10, 6))
plt.plot(x_np, prob_ana, 'k--', label="Analítica", linewidth=2)
plt.plot(x_np, prob_adap, label="PINN Adaptativa", linewidth=2, color='orange')
plt.xlabel("x")
plt.ylabel("|ψ(x)|²")
plt.title(f"Densidad de Probabilidad - Malla Adaptativa - Estado n={n}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"densidad_probabilidad_adaptativa_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 7. Comparación de MSE
labels = ['Función de Onda', 'Densidad de Probabilidad']
mse_fija_vals = [mse_fija, mse_prob_fija]
mse_adap_vals = [mse_adap, mse_prob_adap]

x_pos = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x_pos - width/2, mse_fija_vals, width, label='Malla Fija', alpha=0.8, color='blue')
plt.bar(x_pos + width/2, mse_adap_vals, width, label='Malla Adaptativa', alpha=0.8, color='orange')

plt.xlabel('Tipo de Comparación')
plt.ylabel('MSE')
plt.title(f'Comparación MSE - Estado n={n}')
plt.xticks(x_pos, labels)
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Añadir valores numéricos encima de las barras
for i, (v1, v2) in enumerate(zip(mse_fija_vals, mse_adap_vals)):
    plt.text(i - width/2, v1, f'{v1:.2e}', ha='center', va='bottom')
    plt.text(i + width/2, v2, f'{v2:.2e}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(f"comparacion_mse_n{n}.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Tabla resumen
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)
print(f"Estado cuántico: n = {n}")
print(f"Energía teórica: {n + 0.5:.1f}")
print("-"*60)
print(f"{'Método':<15} {'Tiempo(s)':<10} {'Puntos':<8} {'MSE ψ':<12} {'MSE |ψ|²':<12}")
print("-"*60)
print(f"{'Malla Fija':<15} {tiempo_fija:<10.2f} {puntos_fija:<8} {mse_fija:<12.2e} {mse_prob_fija:<12.2e}")
print(f"{'Malla Adaptativa':<15} {tiempo_adap:<10.2f} {puntos_adap:<8} {mse_adap:<12.2e} {mse_prob_adap:<12.2e}")
print("="*60)


print(f"\nArchivos guardados:")
print(f"- evolucion_energia_fija_n{n}.png")
print(f"- evolucion_energia_adaptativa_n{n}.png")
print(f"- funcion_onda_fija_n{n}.png")
print(f"- funcion_onda_adaptativa_n{n}.png")
print(f"- densidad_probabilidad_fija_n{n}.png")
print(f"- densidad_probabilidad_adaptativa_n{n}.png")
print(f"- comparacion_mse_n{n}.png")
print(f"- resumen_comparacion_n{n}.txt")
