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
    energy_ref=-0.5
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

    mse_psi = np.mean((u_pred_norm - u_analytical_norm)**2)
    psi2_pred = np.abs(u_pred_norm)**2
    psi2_analytical = u_analytical_norm**2
    mse_psi2 = np.mean((psi2_pred - psi2_analytical)**2)

    print(f"\n--- {label} ---")
    print(f"Energía final PINN: {energy:.6f}")
    print(f"MSE función de onda: {mse_psi:.3e}")
    print(f"MSE densidad de probabilidad: {mse_psi2:.3e}")

    # Gráfica función de onda (individual)
    plt.figure(figsize=(7, 5))
    plt.plot(r_eval, u_pred_norm, label=f"PINN {label}")
    plt.plot(r_eval, u_analytical_norm, '--', color="k", label="Analítica")
    plt.xlabel("r")
    plt.ylabel("Función de onda")
    plt.title(f"Función de onda: {label}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"wave_comparison_{label.lower()}.png")
    plt.close()

    # Gráfica densidad de probabilidad (individual)
    plt.figure(figsize=(7, 5))
    plt.plot(r_eval, psi2_pred, label=f"PINN {label}")
    plt.plot(r_eval, psi2_analytical, '--', color="k", label="Analítica")
    plt.xlabel("r")
    plt.ylabel("Densidad de probabilidad")
    plt.title(f"Densidad de probabilidad: {label}")
    plt.legend()
    plt.grid()
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
        label="Adaptativa"
    )
    # Fija
    eval_model(
        model_path="modelo_FINAL_hidrogeno_fija.pth",
        E_history_path="E_history_hidrogeno_fija.npy",
        label="Fija"
    )
