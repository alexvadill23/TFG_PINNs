import torch
import numpy as np
import matplotlib.pyplot as plt
from pinn_TDSE import SchrodingerPINN_TDSE

L = 6.0
T_total = 2.0
num_x = 200
num_t = 50
batch_size = 2000
num_epochs = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def V(x):
    return 0.5 * x**2

def psi0_superposicion(x):
    x_np = x.cpu().numpy().squeeze()
    psi0_0 = (1/np.pi**0.25) * np.exp(-x_np**2 / 2)
    psi0_1 = (np.sqrt(2)/np.pi**0.25) * x_np * np.exp(-x_np**2 / 2)
    psi = psi0_0 + psi0_1
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, x_np))
    if norm == 0:
        return psi
    return psi / norm

def get_psi0_from_prev_model(model, x, t_prev):
    xt = torch.cat([x, torch.full_like(x, t_prev)], dim=1)
    with torch.no_grad():
        psi = model(xt)
    psi_real = psi[:, 0].cpu().numpy()
    psi_imag = psi[:, 1].cpu().numpy()
    return psi_real, psi_imag

def initial_condition_loss(model, x0, psi0_real, psi0_imag):
    t0 = torch.zeros_like(x0)
    xt0 = torch.cat([x0, t0], dim=1)
    psi = model(xt0)
    psi_real = psi[:, 0]
    psi_imag = psi[:, 1]
    psi0_real_torch = torch.tensor(psi0_real, dtype=torch.float32, device=x0.device)
    psi0_imag_torch = torch.tensor(psi0_imag, dtype=torch.float32, device=x0.device)
    loss_ic = torch.mean((psi_real - psi0_real_torch)**2 + (psi_imag - psi0_imag_torch)**2)
    return loss_ic

def physics_loss(model, x, t):
    xt = torch.cat([x, t], dim=1)
    psi = model(xt)
    psi_real = psi[:, 0:1]
    psi_imag = psi[:, 1:2]
    dpsi_real_dt = torch.autograd.grad(psi_real, t, grad_outputs=torch.ones_like(psi_real), create_graph=True)[0]
    dpsi_imag_dt = torch.autograd.grad(psi_imag, t, grad_outputs=torch.ones_like(psi_imag), create_graph=True)[0]
    dpsi_real_dx = torch.autograd.grad(psi_real, x, grad_outputs=torch.ones_like(psi_real), create_graph=True)[0]
    dpsi_imag_dx = torch.autograd.grad(psi_imag, x, grad_outputs=torch.ones_like(psi_imag), create_graph=True)[0]
    d2psi_real_dx2 = torch.autograd.grad(dpsi_real_dx, x, grad_outputs=torch.ones_like(dpsi_real_dx), create_graph=True)[0]
    d2psi_imag_dx2 = torch.autograd.grad(dpsi_imag_dx, x, grad_outputs=torch.ones_like(dpsi_imag_dx), create_graph=True)[0]
    Vx = V(x)
    eq_real = dpsi_imag_dt + 0.5 * d2psi_real_dx2 - Vx * psi_real
    eq_imag = -dpsi_real_dt + 0.5 * d2psi_imag_dx2 - Vx * psi_imag
    loss_pde = torch.mean(eq_real**2 + eq_imag**2)
    return loss_pde

def boundary_condition_loss(model, t):
    t_bc = t.repeat(2, 1)
    x_bc = torch.cat([
        torch.full_like(t, -L),
        torch.full_like(t, L)
    ], dim=0)
    xt_bc = torch.cat([x_bc, t_bc], dim=1)
    psi = model(xt_bc)
    loss_bc = torch.mean(psi**2)
    return loss_bc

def normalization_loss(model, x, t):
    loss = 0.0
    t_sample = t
    for ti in t_sample:
        t_col = torch.full_like(x, ti.item())
        xt = torch.cat([x, t_col], dim=1)
        psi = model(xt)
        psi_real = psi[:, 0]
        psi_imag = psi[:, 1]
        prob = psi_real**2 + psi_imag**2
        integral = torch.trapz(prob, x.squeeze())
        loss += (integral - 1.0)**2
    return loss / len(t_sample)

def train_multistage(num_stages):
    T_stage = T_total / num_stages
    x = torch.linspace(-L, L, num_x, device=device).view(-1, 1)
    t = torch.linspace(0, T_stage, num_t, device=device).view(-1, 1)
    x_grid, t_grid = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_flat = x_grid.reshape(-1, 1)
    t_flat = t_grid.reshape(-1, 1)
    x_flat.requires_grad_()
    t_flat.requires_grad_()

    psi0_real = psi0_superposicion(x)
    psi0_imag = np.zeros_like(psi0_real)

    for stage in range(num_stages):
        print(f"\n--- Entrenando etapa {stage+1}/{num_stages}, t en [{stage*T_stage}, {(stage+1)*T_stage}] ---")
        model = SchrodingerPINN_TDSE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        losses_total = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            idx = torch.randint(0, x_flat.shape[0], (batch_size,))
            x_batch = x_flat[idx]
            t_batch = t_flat[idx]
            loss_pde = physics_loss(model, x_batch, t_batch)
            loss_ic = initial_condition_loss(model, x, psi0_real, psi0_imag)
            loss_bc = boundary_condition_loss(model, t)
            loss_norm = normalization_loss(model, x, t)
            loss = 1.0 * loss_pde + 20.0 * loss_ic + 1.0 * loss_bc + 5.0 * loss_norm
            loss.backward()
            optimizer.step()
            losses_total.append(loss.item())

            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

        torch.save(model.state_dict(), f"modelo_tdse_superposicion_{num_stages}etapas_stage{stage+1}.pth")
        np.save(f"x_vals_stage{stage+1}.npy", x.cpu().numpy())
        np.save(f"psi0_real_stage{stage+1}.npy", psi0_real)
        np.save(f"psi0_imag_stage{stage+1}.npy", psi0_imag)
        np.save(f"losses_stage{stage+1}.npy", np.array(losses_total))
        np.save(f"t_vals_stage{stage+1}.npy", t.cpu().numpy())
        print(f"Modelo guardado: modelo_tdse_superposicion_{num_stages}etapas_stage{stage+1}.pth")

        # Para la siguiente etapa, la CI es la solución final de esta etapa en t = T_stage
        if stage < num_stages - 1:
            model.eval()
            psi0_real, psi0_imag = get_psi0_from_prev_model(model, x, t_prev=T_stage)

    print(f"Entrenamiento por etapas ({num_stages} etapas) completado.")

if __name__ == "__main__":
    for num_stages in [1, 2, 4, 8]:
        train_multistage(num_stages)
