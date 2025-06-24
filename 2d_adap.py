import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_2d import SchrodingerPINN2D
import os
import time
from scipy.stats import linregress
from scipy.interpolate import griddata

L = 6.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_adap_2d(
    n_x, n_y,
    plot_result=True,
):
    print(f"\n=== ENTRENAR ESTADO n=({n_x},{n_y}) (MALLA ADAPTATIVA 2D) ===")
    torch.manual_seed(42)
    np.random.seed(42)
    puntos_iniciales = 400
    puntos_por_refinamiento = 100
    frecuencia_refinamiento = 1000
    max_total_points = 2500
    lr_switched = False

    MARGEN_TOLERANCIA = 0.02
    EPOCAS_ESTABLES_CONSECUTIVAS = 1000
    VENTANA_ESTABILIDAD = 1000

    model = SchrodingerPINN2D().to(device)
    model.E.data = torch.tensor(0, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    # Malla inicial uniforme en 2D
    x = torch.linspace(-L, L, int(np.sqrt(puntos_iniciales)), device=device)
    y = torch.linspace(-L, L, int(np.sqrt(puntos_iniciales)), device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    x_train = torch.stack([X.flatten(), Y.flatten()], dim=1)

    E_history = []
    loss_history = []
    contador_estable = 0
    epoch = 0

    start = time.time()

    while True:
        for _ in range(frecuencia_refinamiento):
            optimizer.zero_grad()
            x_train.requires_grad_(True)
            psi = model(x_train)
            grad = torch.autograd.grad(psi, x_train, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
            d2psi_dx2 = torch.autograd.grad(grad[:,0], x_train, grad_outputs=torch.ones_like(grad[:,0]), create_graph=True)[0][:,0]
            d2psi_dy2 = torch.autograd.grad(grad[:,1], x_train, grad_outputs=torch.ones_like(grad[:,1]), create_graph=True)[0][:,1]
            H_psi = -0.5 * (d2psi_dx2 + d2psi_dy2) + 0.5 * (x_train[:,0]**2 + x_train[:,1]**2) * psi.squeeze()
            loss_pde = torch.mean((H_psi - model.E * psi.squeeze())**2)
            # Condiciones de contorno en el borde del cuadrado
            mask_bc = ((torch.abs(x_train[:,0]) == L) | (torch.abs(x_train[:,1]) == L))
            loss_bc = torch.mean(psi[mask_bc]**2) if mask_bc.any() else torch.tensor(0.0, device=device)
            # Normalización 2D
            prob_density = psi.squeeze()**2
            area = (2*L)*(2*L) / x_train.shape[0]
            integral = torch.sum(prob_density) * area
            loss_norm = (integral - 1)**2
            # Simetría respecto a cada eje (paridad)
            loss_sym = torch.tensor(0.0, device=device)
            for i, n_i in enumerate([n_x, n_y]):
                x_flip = x_train.clone()
                x_flip[:, i] *= -1
                psi_flip = model(x_flip)
                if n_i % 2 == 0:
                    loss_sym += torch.mean((psi - psi_flip)**2)
                else:
                    loss_sym += torch.mean((psi + psi_flip)**2)

            loss_proj = torch.tensor(0.0, device=device)
            for kx in range(n_x+1):
                for ky in range(n_y+1):
                    if (kx, ky) == (n_x, n_y):
                        continue

                    fname_psi = f"psi_pred_n{kx}_{ky}_2d_adap.npy"
                    fname_xy = f"x_vals_n{kx}_{ky}_2d_adap.npy"
                    if os.path.exists(fname_psi) and os.path.exists(fname_xy):
                        psi_k = np.load(fname_psi)
                        xy_k = np.load(fname_xy)
                        xy_current = x_train.detach().cpu().numpy()
                        # Interpolación de psi_k a los puntos actuales
                        psi_k_interp = griddata(xy_k, psi_k, xy_current, method='linear', fill_value=0.0)
                        psi_k_tensor = torch.tensor(psi_k_interp, dtype=torch.float32, device=device).view(-1)
                        psi_actual = psi.squeeze()
                        # Integral doble (aprox.) usando suma y área por punto
                        area = (2*L)*(2*L) / xy_current.shape[0]
                        overlap = torch.sum(psi_k_tensor * psi_actual) * area
                        norm_k = torch.sqrt(torch.sum(psi_k_tensor**2) * area)
                        norm_actual = torch.sqrt(torch.sum(psi_actual**2) * area)
                        overlap_norm = overlap / (norm_k * norm_actual + 1e-12)

            total_loss = (
                0.3 * loss_pde +
                1.0 * loss_bc +
                3.0 * loss_norm +
                5.0 * loss_sym +
                10.0 * loss_proj
            )
            if len(E_history) >= 10:
                recent_E = torch.tensor(E_history[-10:], device=device)
                energy_var = torch.var(recent_E)
                total_loss += 0.03 * energy_var
            total_loss.backward(retain_graph=True)
            optimizer.step()

            current_energy = model.E.item()
            E_history.append(current_energy)
            loss_history.append(total_loss.item())

            # Criterio de parada: estabilidad + pendiente
            if len(E_history) >= VENTANA_ESTABILIDAD:
                energias_recientes = E_history[-VENTANA_ESTABILIDAD:]
                energia_media = np.mean(energias_recientes)
                desviacion = np.std(energias_recientes)
                variacion_relativa = desviacion / abs(energia_media) if abs(energia_media) > 1e-8 else float('inf')
                epochs_window = np.arange(VENTANA_ESTABILIDAD)
                slope, _, _, _, _ = linregress(epochs_window, energias_recientes)
                if variacion_relativa < MARGEN_TOLERANCIA and abs(slope) < 1e-5:
                    contador_estable += 1
                else:
                    contador_estable = 0
                if contador_estable >= EPOCAS_ESTABLES_CONSECUTIVAS:
                    print(f"\nCriterio de estabilidad alcanzado: {contador_estable} épocas estables consecutivas (variación relativa={variacion_relativa:.6f}, pendiente={slope:.2e})")
                    break

            if epoch % 1000 == 0:
                print(f"Época {epoch}: Loss={total_loss.item():.4f}, E={current_energy:.6f}, Puntos={x_train.shape[0]}")
            epoch += 1

        # Refinamiento adaptativo
        if x_train.shape[0] < max_total_points:
            x_dense = torch.linspace(-L, L, 50, device=device)
            y_dense = torch.linspace(-L, L, 50, device=device)
            Xd, Yd = torch.meshgrid(x_dense, y_dense, indexing='ij')
            x_dense_grid = torch.stack([Xd.flatten(), Yd.flatten()], dim=1)
            x_dense_grid.requires_grad_(True)
            psi_dense = model(x_dense_grid)
            grad_dense = torch.autograd.grad(psi_dense, x_dense_grid, grad_outputs=torch.ones_like(psi_dense), create_graph=True)[0]
            d2psi_dx2_dense = torch.autograd.grad(grad_dense[:,0], x_dense_grid, grad_outputs=torch.ones_like(grad_dense[:,0]), create_graph=True)[0][:,0]
            d2psi_dy2_dense = torch.autograd.grad(grad_dense[:,1], x_dense_grid, grad_outputs=torch.ones_like(grad_dense[:,1]), create_graph=True)[0][:,1]
            H_psi_dense = -0.5 * (d2psi_dx2_dense + d2psi_dy2_dense) + 0.5 * (x_dense_grid[:,0]**2 + x_dense_grid[:,1]**2) * psi_dense.squeeze()
            residuo = torch.abs(H_psi_dense - model.E * psi_dense.squeeze()).detach().cpu().numpy().flatten()
            idx_top = np.argsort(residuo)[-puntos_por_refinamiento:]
            nuevos_puntos = x_dense_grid[idx_top]
            x_train = torch.cat([x_train, nuevos_puntos], dim=0)
            print(f"Refinamiento: +{puntos_por_refinamiento} puntos añadidos, total: {x_train.shape[0]}")
        elif not lr_switched:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-4
            lr_switched = True
            print(f"Learning rate cambiado a 1e-4 (malla completa alcanzada)")

        if contador_estable >= EPOCAS_ESTABLES_CONSECUTIVAS:
            break

    end = time.time()

    # Guardar resultados
    torch.save(model.state_dict(), f"modelo_n{n_x}_{n_y}_2d_adap.pth")
    training_info = {
        'training_time': end - start,
    }
    model.eval()
    with torch.no_grad():
        psi_pred = model(x_train).detach().cpu().numpy()
    x_np = x_train.detach().cpu().numpy()
    area = (2*L)*(2*L) / x_np.shape[0]
    norm = np.sqrt(np.sum(np.abs(psi_pred.squeeze())**2) * area)
    psi_pred_norm = psi_pred / norm
    np.save(f"x_vals_n{n_x}_{n_y}_2d_adap.npy", x_np)
    np.save(f"psi_pred_n{n_x}_{n_y}_2d_adap.npy", psi_pred_norm.squeeze())
    np.save(f"energia_history_n{n_x}_{n_y}_2d_adap.npy", np.array(E_history))
    np.save(f"loss_history_n{n_x}_{n_y}_2d_adap.npy", np.array(loss_history))
    np.save(f"training_info_n{n_x}_{n_y}_2d_adap.npy", training_info)

if __name__ == "__main__":
    n_x = int(input("Introduce el número cuántico n_x (0,1,2,...): "))
    n_y = int(input("Introduce el número cuántico n_y (0,1,2,...): "))
    train_adap_2d(n_x, n_y)