import torch
import matplotlib.pyplot as plt
import numpy as np
from pinn import SchrodingerPINN
import time
import random

# Semilla para reproducibilidad
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parámetros del problema
r_min = 1e-3
r_max = 10.0
num_points_fixed = 2500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_energy = -0.5
max_epochs = 1000000

# Potencial suavizado
epsilon = 0.05  

def physics_loss(model, r):
    r.requires_grad_(True)
    u = model(r)
    du_dr = torch.autograd.grad(u, r, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dr2 = torch.autograd.grad(du_dr, r, grad_outputs=torch.ones_like(du_dr), create_graph=True)[0]
    mask_physics = r.squeeze() <= 7.0
    mask_asymptotic = r.squeeze() > 7.0

    if mask_physics.any():
        u_phys = u[mask_physics.unsqueeze(1)]
        r_phys = r[mask_physics.unsqueeze(1)]
        d2u_dr2_phys = d2u_dr2[mask_physics.unsqueeze(1)]
        V_phys = -1.0 / torch.sqrt(r_phys**2 + epsilon**2)
        H_u = -0.5 * d2u_dr2_phys + V_phys * u_phys
        loss_pde = torch.mean((H_u - model.E * u_phys)**2)
        prob_density = u_phys.squeeze()**2
        integral = torch.trapz(prob_density, r_phys.squeeze())
        loss_norm = (integral - 1)**2
    else:
        loss_pde = torch.tensor(0.0, device=r.device)
        loss_norm = torch.tensor(0.0, device=r.device)

    r_bc = torch.tensor([[r_min]], dtype=torch.float32, device=r.device)
    u_bc = model(r_bc)
    loss_bc = torch.mean(u_bc**2)

    if mask_asymptotic.any():
        u_asymptotic = u[mask_asymptotic.unsqueeze(1)]
        loss_asymptotic = torch.mean(u_asymptotic**2)
    else:
        loss_asymptotic = torch.tensor(0.0, device=r.device)

    total_loss = 2.0 * loss_pde + 1 * loss_bc + 3 * loss_norm + 2.5 * loss_asymptotic
    return total_loss, loss_pde

def train_model_fija():
    model = SchrodingerPINN().to(device)
    model.E.data = torch.tensor([0.0], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    r_train = torch.linspace(r_min, r_max, num_points_fixed, device=device).view(-1, 1)
    E_history, loss_history, num_points_history = [], [], []

    # --- Criterio de parada energético ---
    window = 1000
    tol_std = 0.0001
    tol_slope = 1e-6
    early_stop = False

    start = time.time()

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss, loss_pde = physics_loss(model, r_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        current_energy = model.E.item()
        E_history.append(current_energy)
        loss_history.append(loss.item())
        num_points_history.append(r_train.shape[0])

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}, E = {current_energy:.6f}, puntos = {r_train.shape[0]}")

        # --- Criterio de parada por estabilización de la energía ---
        if epoch > window:
            E_window = np.array(E_history[-window:])
            std = np.std(E_window)
            x = np.arange(window)
            slope = np.polyfit(x, E_window, 1)[0]
            if std < tol_std and abs(slope) < tol_slope:
                print(f"\nCriterio de parada alcanzado en epoch {epoch}: std energía={std:.2e}, pendiente={slope:.2e}")
                early_stop = True
                break


    end = time.time()
    tiempo_total = end - start
    print(f"\n ENTRENAMIENTO COMPLETADO: {tiempo_total:.2f} segundos")
    print(f" Puntos finales: {r_train.shape[0]}")

    # Guardar el modelo al final del entrenamiento
    torch.save(model.state_dict(), "modelo_FINAL_hidrogeno_fija.pth")
    print(f" Modelo guardado: modelo_FINAL_hidrogeno_fija.pth")

    # Evaluación final
    model.eval()
    with torch.no_grad():
        u_pred = model(r_train).detach().cpu().numpy()

    r_np = r_train.detach().cpu().numpy().squeeze()
    mask_phys_np = r_np <= 7.0
    r_phys_np = r_np[mask_phys_np]
    u_phys_np = u_pred.squeeze()[mask_phys_np]
    norm = np.trapezoid(np.abs(u_phys_np)**2, r_phys_np)
    if norm > 0:
        u_pred_norm = u_pred / np.sqrt(norm)
    else:
        u_pred_norm = u_pred

    u_analytical = 2 * r_phys_np * np.exp(-r_phys_np)
    norm_analytical = np.trapezoid(u_analytical**2, r_phys_np)
    if norm_analytical > 0:
        u_analytical = u_analytical / np.sqrt(norm_analytical)

    if r_phys_np.size == 0:
        mse = mae = max_error = float('nan')
    else:
        mse = np.mean((u_pred_norm.squeeze()[mask_phys_np] - u_analytical)**2)
        mae = np.mean(np.abs(u_pred_norm.squeeze()[mask_phys_np] - u_analytical))
        max_error = np.max(np.abs(u_pred_norm.squeeze()[mask_phys_np] - u_analytical))
    energy_error_final = abs(model.E.item() - target_energy)

    resultados = {
        'tiempo_total': tiempo_total,
        'energia_final': model.E.item(),
        'error_energia': energy_error_final,
        'mse_funcion': mse,
        'mae_funcion': mae,
        'error_maximo': max_error,
        'puntos_finales': r_train.shape[0],
        'loss_final': loss_history[-1],
        'target_energy': target_energy,
        'criterio_parada': f"std<{tol_std}, slope<{tol_slope}, window={window}" if early_stop else f"max_epochs={max_epochs}",
    }

    np.save("r_vals_hidrogeno_fija.npy", r_np)
    np.save("u_pred_hidrogeno_fija.npy", u_pred_norm.squeeze())
    np.save("loss_history_hidrogeno_fija.npy", np.array(loss_history))
    np.save("E_history_hidrogeno_fija.npy", np.array(E_history))
    np.save("num_points_history_hidrogeno_fija.npy", np.array(num_points_history))
    np.save("resultados_entrenamiento_hidrogeno_fija.npy", resultados)

    print(f"\n RESULTADOS FINALES:")
    print(f"   Tiempo total: {tiempo_total:.2f} segundos")
    print(f"   Energía final: {model.E.item():.6f}")
    print(f"   Error energía vs -0.5: {energy_error_final:.6f}")
    print(f"   MSE función: {mse:.3e}")
    print(f"   MAE función: {mae:.3e}")
    print(f"   Error máximo: {max_error:.3e}")
    print(f"   Puntos finales: {r_train.shape[0]}")


if __name__ == "__main__":
    print(" Entrenando PINN para el átomo de hidrógeno 1d (malla fija)...")
    train_model_fija()
