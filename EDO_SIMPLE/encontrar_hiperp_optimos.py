import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Definición del modelo PINN
class PINN(nn.Module):
    def __init__(self, n_layers=2, neurons=64, activation='relu'):
        super(PINN, self).__init__()
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError("Unknown activation function")

        layers = []
        in_features = 1
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, neurons))
            layers.append(activation_fn)
            in_features = neurons
        self.hidden = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_features, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, x):
        x = self.hidden(x)
        x = self.fc_out(x)
        return x

# Función para calcular la pérdida PDE
def pde_loss(model, X_std, scaler_X):
    sigma_x = scaler_X.scale_[0]
    device = next(model.parameters()).device  
    
    X_tensor = torch.from_numpy(X_std).float().to(device).requires_grad_(True)
    y_std = model(X_tensor)
    
    dy_dx = torch.autograd.grad(outputs=y_std, inputs=X_tensor,
                                grad_outputs=torch.ones_like(y_std, device=device),
                                create_graph=True, retain_graph=True)[0]
    
    d2y_dx2 = torch.autograd.grad(outputs=dy_dx, inputs=X_tensor,
                                  grad_outputs=torch.ones_like(dy_dx, device=device),
                                  create_graph=True, retain_graph=True)[0]
    
    residual = (1 / (sigma_x ** 2)) * d2y_dx2 + y_std
    loss_pde = torch.mean(residual ** 2)
    
    return loss_pde

# Clase del modelo con validación
class PINNModel:
    def __init__(self, n_layers=2, neurons=64, activation='relu', lr=0.001, epochs=5000, patience=50):
        self.n_layers = n_layers
        self.neurons = neurons
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.model = None
        self.optimizer = None
        self.loss_fn = nn.MSELoss()
    
    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32, device=device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32, device=device)
        
        self.model = PINN(n_layers=self.n_layers, neurons=self.neurons, activation=self.activation).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_loss = float('inf')
        epochs_without_improvement = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            y_pred_train = self.model(X_train_tensor)
            data_loss_train = self.loss_fn(y_pred_train, y_train_tensor)
            pde_loss_train = pde_loss(self.model, X_train_scaled, self.scaler_X)
            train_loss = data_loss_train + pde_loss_train
            train_loss.backward()
            self.optimizer.step()
            
            # Validación: Se calcula la data loss sin gradientes
            self.model.eval()
            with torch.no_grad():
                y_pred_val = self.model(X_val_tensor)
                data_loss_val = self.loss_fn(y_pred_val, y_val_tensor)
            # Se habilitan gradientes solo para calcular pde_loss
            with torch.enable_grad():
                pde_loss_val = pde_loss(self.model, X_val_scaled, self.scaler_X)
            val_loss = data_loss_val + pde_loss_val

            if epoch % 100 == 0:
                print(f"Epoch {epoch}/{self.epochs}: Train Loss = {train_loss.item():.4f}, Val Loss = {val_loss.item():.4f}")

            tolerance = 1e-4  # Solo cuenta como mejora si el cambio es mayor a esto

            if best_loss - val_loss.item() > tolerance:  
                best_loss = val_loss.item()  
                epochs_without_improvement = 0  
            else:  
                epochs_without_improvement += 1  

            if epochs_without_improvement >= 50:  
                print(f"Detenido en la época {epoch} por falta de mejora en {epochs_without_improvement} épocas.")
                break

        return train_loss.item(), val_loss.item()
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        device = next(self.model.parameters()).device
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred_std = self.model(X_tensor)
        y_pred = self.scaler_y.inverse_transform(y_pred_std.cpu().numpy())
        return y_pred.flatten()

# Definición del grid de hiperparámetros
param_grid = {
    'activation': ['relu', 'tanh'],
    'n_layers': [2, 3, 4, 5],
    'neurons': [32, 64, 128],
    'lr': [0.001, 0.01, 0.1]
}

X = np.linspace(0, 4*np.pi, 100).reshape(-1, 1)
y = np.cos(X).flatten()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

results = []
for activation in param_grid['activation']:
    for n_layers in param_grid['n_layers']:
        for neurons in param_grid['neurons']:
            for lr in param_grid['lr']:
                print(f"\nProbando: activation={activation}, n_layers={n_layers}, neurons={neurons}, lr={lr}")
                # Aquí se establece epochs=5000 y se ajusta patience para que no se active early stopping
                model = PINNModel(n_layers=n_layers, neurons=neurons, activation=activation, lr=lr, epochs=5000, patience=5000)
                train_loss, val_loss = model.fit(X_train, y_train, X_val, y_val)
                
                results.append({'activation': activation, 'n_layers': n_layers, 'neurons': neurons, 'lr': lr, 'train_loss': train_loss, 'val_loss': val_loss})

results_df = pd.DataFrame(results)
results_df.to_csv('hyperparameter_search_results.csv', index=False)
print("\nBúsqueda completada. Resultados guardados en 'hyperparameter_search_results.csv'.")
