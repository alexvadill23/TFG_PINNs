import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_2d import SchrodingerPINN2D
import math
from scipy.special import hermite

L = 6.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psi_analitica_2d(n_x, n_y, x, y):
    Hx = hermite(n_x)
    Hy = hermite(n_y)
    norm_x = 1.0 / np.sqrt((2.0**n_x) * math.factorial(n_x) * np.sqrt(np.pi))
    norm_y = 1.0 / np.sqrt((2.0**n_y) * math.factorial(n_y) * np.sqrt(np.pi))
    return norm_x * norm_y * np.exp(-0.5*(x**2 + y**2)) * Hx(x) * Hy(y)

def eval_model_2d(model_path, x_path, n_x, n_y):
    model = SchrodingerPINN2D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    n_plot = 100
    x_plot = np.linspace(-L, L, n_plot)
    y_plot = np.linspace(-L, L, n_plot)
    Xp, Yp = np.meshgrid(x_plot, y_plot, indexing='ij')
    xy_plot = torch.tensor(np.stack([Xp.flatten(), Yp.flatten()], axis=1), dtype=torch.float32, device=device)
    with torch.no_grad():
        psi_pred = model(xy_plot).cpu().numpy().reshape(n_plot, n_plot)
    norm_pred = np.sqrt(np.trapezoid(np.trapezoid(np.abs(psi_pred)**2, x_plot, axis=0), y_plot, axis=0))
    psi_pred /= norm_pred
    psi_ana = psi_analitica_2d(n_x, n_y, Xp, Yp)
    return psi_pred, psi_ana, x_plot, y_plot

def save_energy_fig(energy, energy_teo, filename):
    plt.figure()
    plt.plot(energy)
    plt.axhline(energy_teo, color='r', linestyle='--', label='Teórica')
    plt.xlabel("Época")
    plt.ylabel("Energía")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_comparativa_2d_fig(psi_pinn, psi_ana, x_plot, y_plot, filename, tipo="onda", titulo=""):
    plt.figure(figsize=(10,4))
    # Panel izquierdo: PINN
    plt.subplot(1,2,1)
    if tipo == "onda":
        vmin = min(np.min(psi_pinn), np.min(psi_ana))
        vmax = max(np.max(psi_pinn), np.max(psi_ana))
        cmap = 'RdBu'
        plt.contourf(x_plot, y_plot, psi_pinn, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Función de onda')
        plt.title('PINN')
    else:
        vmin = min(np.min(psi_pinn**2), np.min(psi_ana**2))
        vmax = max(np.max(psi_pinn**2), np.max(psi_ana**2))
        cmap = 'viridis'
        plt.contourf(x_plot, y_plot, psi_pinn**2, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Densidad de probabilidad')
        plt.title('PINN')
    plt.xlabel('x')
    plt.ylabel('y')
    # Panel derecho: Analítica
    plt.subplot(1,2,2)
    if tipo == "onda":
        plt.contourf(x_plot, y_plot, psi_ana, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Función de onda')
        plt.title('Analítica')
    else:
        plt.contourf(x_plot, y_plot, psi_ana**2, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label='Densidad de probabilidad')
        plt.title('Analítica')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.suptitle(titulo)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=200)
    plt.close()

def save_corte_y0_fig(psi_pinn, psi_ana, x_plot, y_plot, filename, titulo):
    idx_y0 = np.argmin(np.abs(y_plot))
    plt.figure()
    plt.plot(x_plot, psi_pinn[:, idx_y0], label='PINN')
    plt.plot(x_plot, psi_ana[:, idx_y0], '--', label='Analítica')
    plt.xlabel('x')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def save_corte_x0_fig(psi_pinn, psi_ana, x_plot, y_plot, filename, titulo):
    idx_x0 = np.argmin(np.abs(x_plot))
    plt.figure()
    plt.plot(y_plot, psi_pinn[idx_x0, :], label='PINN')
    plt.plot(y_plot, psi_ana[idx_x0, :], '--', label='Analítica')
    plt.xlabel('y')
    plt.title(titulo)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

if __name__ == "__main__":
    n_x = int(input("Introduce el número cuántico nx (0,1,2,...): "))
    n_y = int(input("Introduce el número cuántico ny (0,1,2,...): "))

    # Adaptativa
    psi_pred_adap, psi_ana_adap, x_plot, y_plot = eval_model_2d(
        model_path=f"modelo_n{n_x}_{n_y}_2d_adaptativo.pth",
        x_path=f"x_vals_n{n_x}_{n_y}_2d_adaptativo.npy",
        n_x=n_x, n_y=n_y
    )
    energia_adap = np.load(f"energia_history_n{n_x}_{n_y}_2d_adaptativo.npy")
    save_energy_fig(energia_adap, n_x+n_y+1.0, f"evolucion_energia_adaptativa_2d_n{n_x}.png")
    save_comparativa_2d_fig(
        psi_pred_adap, psi_ana_adap, x_plot, y_plot,
        f"funcion_onda_adaptativa_2d_n{n_x}.png", tipo="onda",
        titulo="Función de onda 2D: PINN vs Analítica (Adaptativa)"
    )
    save_comparativa_2d_fig(
        psi_pred_adap, psi_ana_adap, x_plot, y_plot,
        f"densidad_probabilidad_adaptativa_2d_n{n_x}.png", tipo="prob",
        titulo="Densidad de probabilidad 2D: PINN vs Analítica (Adaptativa)"
    )
    save_corte_y0_fig(psi_pred_adap, psi_ana_adap, x_plot, y_plot,
                      f"corte_y0_malla_adaptativa_2d_n{n_x}.png", "Corte y=0 malla adaptativa")
    save_corte_x0_fig(psi_pred_adap, psi_ana_adap, x_plot, y_plot,
                      f"corte_x0_malla_adaptativa_2d_n{n_x}.png", "Corte x=0 malla adaptativa")

    # Fija
    psi_pred_fija, psi_ana_fija, x_plot, y_plot = eval_model_2d(
        model_path=f"modelo_n{n_x}_{n_y}_2d_fija.pth",
        x_path=f"x_vals_n{n_x}_{n_y}_2d_fija.npy",
        n_x=n_x, n_y=n_y
    )
    energia_fija = np.load(f"energia_history_n{n_x}_{n_y}_2d_fija.npy")
    save_energy_fig(energia_fija, n_x+n_y+1.0, f"evolucion_energia_fija_2d_n{n_x}.png")
    save_comparativa_2d_fig(
        psi_pred_fija, psi_ana_fija, x_plot, y_plot,
        f"funcion_onda_fija_2d_n{n_x}.png", tipo="onda",
        titulo="Función de onda 2D: PINN vs Analítica (Fija)"
    )
    save_comparativa_2d_fig(
        psi_pred_fija, psi_ana_fija, x_plot, y_plot,
        f"densidad_probabilidad_fija_2d_n{n_x}.png", tipo="prob",
        titulo="Densidad de probabilidad 2D: PINN vs Analítica (Fija)"
    )
    save_corte_y0_fig(psi_pred_fija, psi_ana_fija, x_plot, y_plot,
                      f"corte_y0_malla_fija_2d_n{n_x}.png", "Corte y=0 malla fija")
    save_corte_x0_fig(psi_pred_fija, psi_ana_fija, x_plot, y_plot,
                      f"corte_x0_malla_fija_2d_n{n_x}.png", "Corte x=0 malla fija")

    # --- MÉTRICAS Y PRINT FINAL ---
    puntos_fija = len(np.load(f"x_vals_n{n_x}_{n_y}_2d_fija.npy"))
    puntos_adap = len(np.load(f"x_vals_n{n_x}_{n_y}_2d_adaptativo.npy"))

    E_final_fija = energia_fija[-1]
    E_final_adap = energia_adap[-1]
    E_teorica = n_x + n_y + 1.0

    error_fija = abs(E_final_fija - E_teorica) / E_teorica * 100
    error_adap = abs(E_final_adap - E_teorica) / E_teorica * 100
    tiempo_fija = np.load(f"tiempo_n{n_x}_{n_y}_2d_fija.npy")[0]
    tiempo_adap = np.load(f"tiempo_entrenamiento_n{n_x}_{n_y}_2d_adaptativo.npy")[0]

    mse_fija = np.mean((psi_pred_fija - psi_ana_fija)**2)
    mse_adap = np.mean((psi_pred_adap - psi_ana_adap)**2)
    prob_fija = psi_pred_fija**2
    prob_adap = psi_pred_adap**2
    prob_ana = psi_ana_fija**2
    mse_prob_fija = np.mean((prob_fija - prob_ana)**2)
    mse_prob_adap = np.mean((prob_adap - prob_ana)**2)

    print(f"- Tiempo de entrenamiento - Fija: {tiempo_fija:.2f} s")
    print(f"- Tiempo de entrenamiento - Adaptativa: {tiempo_adap:.2f} s")
    print(f"- Energía final - Fija: {E_final_fija:.6f} (error: {error_fija:.2f}%)")
    print(f"- Energía final - Adaptativa: {E_final_adap:.6f} (error: {error_adap:.2f}%)")
    print(f"- Puntos de entrenamiento - Fija: {puntos_fija}, Adaptativa: {puntos_adap}")
    print(f"- MSE función de onda - Fija: {mse_fija:.2e}, Adaptativa: {mse_adap:.2e}")
    print(f"- MSE densidad probabilidad - Fija: {mse_prob_fija:.2e}, Adaptativa: {mse_prob_adap:.2e}")
