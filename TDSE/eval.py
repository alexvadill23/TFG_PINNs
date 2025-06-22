import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_TDSE import SchrodingerPINN_TDSE
import matplotlib.animation as animation
import shutil 

L = 6.0
T_total = 2.0
num_x = 200
num_t = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def psi_analitica(x, t):
    X, T = np.meshgrid(x, t, indexing='ij')
    psi0 = (1/np.pi**0.25) * np.exp(-X**2 / 2)
    psi1 = (np.sqrt(2)/np.pi**0.25) * X * np.exp(-X**2 / 2)
    c0 = 1/np.sqrt(2)
    c1 = 1/np.sqrt(2)
    E0 = 0.5
    E1 = 1.5
    psi = c0 * psi0 * np.exp(-1j * E0 * T) + c1 * psi1 * np.exp(-1j * E1 * T)
    return np.abs(psi)**2

x = torch.linspace(-L, L, num_x, device=device).view(-1, 1)
x_np = x.cpu().numpy().squeeze()

for num_stages in [1, 2, 4, 8]:
    T_stage = T_total / num_stages
    t_full = np.linspace(0, T_total, num_stages * num_t)
    psi_mod2_full = np.zeros((num_x, num_stages * num_t))
    psi_analitica_full = np.zeros((num_x, num_stages * num_t))

    for stage in range(num_stages):
        model = SchrodingerPINN_TDSE().to(device)
        model.load_state_dict(torch.load(f"modelo_tdse_superposicion_{num_stages}etapas_stage{stage+1}.pth", map_location=device))
        model.eval()

        t_start = stage * T_stage
        t_end = (stage + 1) * T_stage
        t = torch.linspace(0, T_stage, num_t, device=device).view(-1, 1)
        t_np = np.linspace(t_start, t_end, num_t)
        x_grid, t_grid = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
        x_flat = x_grid.reshape(-1, 1)
        t_flat = t_grid.reshape(-1, 1)
        xt = torch.cat([x_flat, t_flat], dim=1)

        with torch.no_grad():
            psi = model(xt)
            psi_real = psi[:, 0].cpu().numpy().reshape(num_x, num_t)
            psi_imag = psi[:, 1].cpu().numpy().reshape(num_x, num_t)
            psi_mod2 = psi_real**2 + psi_imag**2

        psi_mod2_full[:, stage*num_t:(stage+1)*num_t] = psi_mod2
        psi_analitica_full[:, stage*num_t:(stage+1)*num_t] = psi_analitica(x_np, t_np)

    mse_t = np.mean((psi_mod2_full - psi_analitica_full)**2, axis=0)
    mse_medio = np.mean(mse_t)
    print(f"MSE medio para {num_stages} etapas: {mse_medio:.4e}")
    np.save(f"mse_t_{num_stages}etapas.npy", mse_t)
    np.save(f"t_full_{num_stages}etapas.npy", t_full)
    print(f"Guardado mse_t_{num_stages}etapas.npy")

    # 1. Evolución temporal de la densidad de probabilidad para varios tiempos (cada una en su propia gráfica)
    tiempos = [0, int(0.25*len(t_full)), int(0.5*len(t_full)), int(0.75*len(t_full)), -1]
    tiempos_labels = [f"{t_full[idx]:.2f}" for idx in tiempos]
    for idx, t_label in zip(tiempos, tiempos_labels):
        plt.figure()
        plt.plot(x_np, psi_mod2_full[:, idx], label=f'PINN t={t_label}')
        plt.plot(x_np, psi_analitica_full[:, idx], '--', label=f'Analítica t={t_label}')
        plt.xlabel('x')
        plt.ylabel('|ψ(x, t)|²')
        plt.title(f'Comparación de la densidad para t={t_label} ({num_stages} etapa(s))')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"evolucion_temporal_{num_stages}etapas_t{t_label.replace('.','p')}.png")
        plt.close()

    # 2. Mapa de calor (heatmap) espacio-tiempo
    plt.figure(figsize=(8, 4))
    plt.imshow(psi_mod2_full, aspect='auto', origin='lower',
               extent=[t_full[0], t_full[-1], x_np[0], x_np[-1]],
               cmap='viridis')
    plt.colorbar(label='|ψ(x, t)|²')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Heatmap PINN |ψ(x, t)|² ({num_stages} etapa(s))')
    plt.tight_layout()
    plt.savefig(f"heatmap_pinn_{num_stages}etapas.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.imshow(psi_analitica_full, aspect='auto', origin='lower',
               extent=[t_full[0], t_full[-1], x_np[0], x_np[-1]],
               cmap='viridis')
    plt.colorbar(label='|ψ(x, t)|²')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(f'Heatmap Analítica |ψ(x, t)|² ({num_stages} etapa(s))')
    plt.tight_layout()
    plt.savefig(f"heatmap_analitica_{num_stages}etapas.png")
    plt.close()

    # 3. Error cuadrático medio (MSE) a lo largo del tiempo
    plt.figure()
    plt.plot(t_full, mse_t, label=f"{num_stages} etapa(s)")
    plt.xlabel("t")
    plt.ylabel("MSE espacial")
    plt.title(f"MSE vs t ({num_stages} etapa(s))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"mse_vs_t_{num_stages}etapas.png")
    plt.close()

    # 4. Animación de la evolución temporal de la densidad de probabilidad
    fig, ax = plt.subplots()
    line_pinn, = ax.plot([], [], label='PINN')
    line_ana, = ax.plot([], [], '--', label='Analítica')
    ax.set_xlim(x_np[0], x_np[-1])
    ax.set_ylim(0, np.max(psi_analitica_full)*1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('|ψ(x, t)|²')
    ax.set_title(f'Evolución temporal |ψ(x, t)|² ({num_stages} etapa(s))')
    ax.legend()

    def animate(i):
        line_pinn.set_data(x_np, psi_mod2_full[:, i])
        line_ana.set_data(x_np, psi_analitica_full[:, i])
        ax.set_title(f'Evolución temporal |ψ(x, t)|² ({num_stages} etapa(s)), t={t_full[i]:.2f}')
        return line_pinn, line_ana
    
    ani = animation.FuncAnimation(fig, animate, frames=len(t_full), interval=60, blit=True)
    
    if shutil.which("ffmpeg"):
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=15, metadata=dict(artist='PINN'), bitrate=1800)
        ani.save(f'animacion_densidad_{num_stages}etapas.mp4', writer=writer)
        print(f"Animación guardada como MP4 para {num_stages} etapas.")
    else:
        print("ffmpeg no encontrado, guardando animación como GIF.")
        ani.save(f'animacion_densidad_{num_stages}etapas.gif', writer='pillow')
    plt.close()
    

# --- Gráfica conjunta de MSE ---
plt.figure()
for num_stages in [1, 2, 4, 8]:
    mse_t = np.load(f"mse_t_{num_stages}etapas.npy")
    t_full = np.load(f"t_full_{num_stages}etapas.npy")
    plt.plot(t_full, mse_t, label=f"{num_stages} etapa(s)")
plt.xlabel("t")
plt.ylabel("MSE espacial")
plt.legend()
plt.title("Comparación MSE vs t para distintas divisiones por etapas")
plt.tight_layout()
plt.savefig("comparacion_mse_vs_t_etapas.png")
plt.show()
