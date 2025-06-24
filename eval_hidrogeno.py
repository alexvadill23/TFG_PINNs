import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn import SchrodingerPINN

def eval_model(
    model_path,
    E_history_path,
    label,
    r_min=1e-3,
    r_cutoff=7.0,
    epsilon=0.05,
    energy_ref=-0.5,
    r_train_path=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchrodingerPINN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    r_eval = np.linspace(r_min, 10, 500)
    r_tensor = torch.tensor(r_eval, dtype=torch.float32, device=device).view(-1, 1)

    with torch.no_grad():
        u_pred = model(r_tensor).cpu().numpy().squeeze()
        energy = model.E.item()

    norm_pred = np.trapezoid(np.abs(u_pred)**2, r_eval)
    u_pred_norm = u_pred / np.sqrt(norm_pred) if norm_pred > 0 else u_pred

    u_analytical = 2 * r_eval * np.exp(-r_eval)
    norm_analytical = np.trapezoid(u_analytical**2, r_eval)
    u_analytical_norm = u_analytical / np.sqrt(norm_analytical) if norm_analytical > 0 else u_analytical

    mse1 = np.mean((u_pred_norm - u_analytical_norm)**2)
    mse2 = np.mean((-u_pred_norm - u_analytical_norm)**2)
    if mse1 < mse2:
        u_pred_corr = u_pred_norm
        mse_psi = mse1
    else:
        u_pred_corr = -u_pred_norm
        mse_psi = mse2

    psi2_pred = np.abs(u_pred_corr)**2
    psi2_analytical = u_analytical_norm**2
    mse_psi2 = np.mean((psi2_pred - psi2_analytical)**2)

    print(f"\n--- {label} ---")
    print(f"Energía final PINN: {energy:.6f}")
    print(f"MSE función de onda: {mse_psi:.3e}")
    print(f"MSE densidad de probabilidad: {mse_psi2:.3e}")
    # Mostrar tiempo de entrenamiento para ambos métodos si existen los archivos
    try:
        res_fija = np.load("resultados_entrenamiento_hidrogeno_fija.npy", allow_pickle=True).item()
        print(f"Tiempo total de entrenamiento (fija): {res_fija['tiempo_total']:.2f} s")
    except Exception as e:
        print(f"No se pudo cargar el tiempo de la malla fija: {e}")

    try:
        res_adap = np.load("resultados_entrenamiento_hidrogeno_FINAL.npy", allow_pickle=True).item()
        print(f"Tiempo total de entrenamiento (adaptativa): {res_adap['tiempo_total']:.2f} s")
    except Exception as e:
        print(f"No se pudo cargar el tiempo de la malla adaptativa: {e}")

    # Cargar puntos de entrenamiento si se proporciona el path
    r_train = None
    if r_train_path is not None:
        try:
            r_train = np.load(r_train_path)
        except Exception as e:
            print(f"No se pudo cargar r_train para {label}: {e}")

    # Gráfica función de onda (individual) + histograma
    plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    ax1.plot(r_eval, u_pred_corr, label=f"PINN {label}")
    ax1.plot(r_eval, u_analytical_norm, '--', color="k", label="Analítica")
    ax1.set_xlabel("r")
    ax1.set_ylabel("Función de onda")
    ax1.set_title(f"Función de onda: {label}")
    ax1.legend(loc="upper right")
    ax1.grid()
    if r_train is not None:
        ax2 = ax1.twinx()
        counts, bins, _ = ax2.hist(r_train, bins=40, color='gray', alpha=0.3, label="Histograma de puntos")
        ax2.set_ylabel("Nº de puntos por intervalo")
        ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"wave_comparison_{label.lower()}.png")
    plt.close()

    # Gráfica densidad de probabilidad (individual) + histograma
    plt.figure(figsize=(7, 5))
    ax1 = plt.gca()
    ax1.plot(r_eval, psi2_pred, label=f"PINN {label}")
    ax1.plot(r_eval, psi2_analytical, '--', color="k", label="Analítica")
    ax1.set_xlabel("r")
    ax1.set_ylabel("Densidad de probabilidad")
    ax1.set_title(f"Densidad de probabilidad: {label}")
    ax1.legend(loc="upper right")
    ax1.grid()
    if r_train is not None:
        ax2 = ax1.twinx()
        counts, bins, _ = ax2.hist(r_train, bins=40, color='gray', alpha=0.3, label="Histograma de puntos")
        ax2.set_ylabel("Nº de puntos por intervalo")
        ax2.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(f"density_comparison_{label.lower()}.png")
    plt.close()

    # Evolución de la energía (individual)
    try:
        E_history = np.load(E_history_path)
        plt.figure(figsize=(7, 5))
        plt.plot(E_history, label=f"Energía PINN {label}")
        plt.axhline(energy_ref, color='r', linestyle='--', label="Energía exacta")
        plt.xlabel("Época")
        plt.ylabel("Energía")
        plt.title(f"Evolución de la energía durante el entrenamiento: {label}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"E_history_{label.lower()}.png")
        plt.close()
    except Exception as e:
        print(f"No se pudo cargar el historial de energía para {label}: {e}")

if __name__ == "__main__":
    # Adaptativa
    eval_model(
        model_path="modelo_FINAL_hidrogeno.pth",
        E_history_path="E_history_hidrogeno_FINAL.npy",
        label="Adaptativa",
        r_train_path="r_vals_hidrogeno_FINAL.npy"
    )
    # Fija
    eval_model(
        model_path="modelo_FINAL_hidrogeno_fija.pth",
        E_history_path="E_history_hidrogeno_fija.npy",
        label="Fija",
        r_train_path="r_vals_hidrogeno_fija.npy"
    )