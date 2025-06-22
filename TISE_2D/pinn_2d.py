import torch
import torch.nn as nn

# Clase principal para la PINN 2D
class SchrodingerPINN2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.E = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))  # Inicializa en 0.6
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, xy):
        # xy: tensor de tamaño (N, 2) donde xy[:,0]=x y xy[:,1]=y
        return self.net(xy)
