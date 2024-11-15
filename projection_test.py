# Projection/projection_test.py

import torch
from src.solver import ConvexOptimizationSolver  # Assicurati che il percorso sia corretto
from torch import nn

# Definizione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Definizione delle matrici dei vincoli G e h
G = torch.tensor([
    [-1, 0],
    [1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
    [0, 1],
    [0, -1]
], dtype=torch.float32).T.to(device)  # [2,7]

h = torch.tensor([0, 0.8, 1.0, 0.8, 1.0, 0.8, -0.8], dtype=torch.float32).to(device)  # [7]

# Funzione di costo personalizzata
def projection_cost(inputs: torch.Tensor, sol: torch.Tensor) -> torch.Tensor:
    """
    Funzione di costo personalizzata: somma dei quadrati delle differenze tra azioni e soluzioni.
    """
    actions = inputs[..., :2]  # Le prime 2 colonne sono le azioni
    return torch.square(actions - sol).sum(dim=-1)

# Funzione di vincolo di disuguaglianza che restituisce tutti i vincoli contemporaneamente
def projection_constraints(inputs: torch.Tensor, sol: torch.Tensor) -> torch.Tensor:
    """
    Funzione di vincolo di disuguaglianza: (sol @ G) - h >= 0
    Restituisce un tensore [batch_size, 7] dove ogni colonna rappresenta un vincolo.
    """
    return (sol @ G) - h  # [batch_size,7]

# Definizione delle variabili con range dipendenti da altre variabili
variable_ranges_projection = {
    0: (0.0, 1.0),                       # Action 0: [0.0, 10.0]
    1: (-1.0, 1.0),   # Action 1: [0.0, A_value]
    2: (lambda X: X[1], lambda X: X[1] + 50.0),  # P_max: [B_value, B_value + 50.0]
    3: (30.0, 100.0),                      # Q_max: [30.0, 100.0] (indipendente)
    4: (10.0, 50.0),                       # P_plus: [10.0, 50.0] (indipendente)
    5: (5.0, 30.0),                        # Q_plus: [5.0, 30.0] (indipendente)
    6: (20.0, 80.0)                        # P_pots: [20.0, 80.0] (indipendente)
}

# Percorso al dataset di validazione
validation_dataset_path = "Projection/projection.pkl"

# Inizializza il solver con generazione di Sobol per l'addestramento e dataset pickle per la validazione
solver_projection = ConvexOptimizationSolver(
    input_dim=7,
    shared_layers=[512, 512],
    sol_layers=[512, 512],
    lambda_layers=[512, 512],
    output_dim_sol=2,  # Soluzione di dimensione 2
    output_dim_lambda=7,  # Moltiplicatori di Lagrange per 7 vincoli
    activation_sol=nn.Sigmoid(),  # Ad esempio, sigmoid per la soluzione
    cost_function=projection_cost,
    inequality_constraints=projection_constraints,  # Funzione unica che restituisce [batch_size,7]
    action_indices=[0, 1],  # Indici delle azioni nelle inputs
    variable_ranges=variable_ranges_projection,
    dataset_path=validation_dataset_path,  # Percorso al dataset di validazione
    num_samples=100000,  # Numero di campioni da generare tramite Sobol per l'addestramento
    batch_size=512,
    learning_rate=3e-4,
    patience=1000,
    optimizer_type='adam',
    scheduler_type='reduce_on_plateau',
    device=device  # Passa il dispositivo al solver
)

# Addestra il modello
solver_projection.train_model(epochs=10)

# Salva i log e il modello
solver_projection.save_logs(log_path="Projection/log.csv", metrics_path="Projection/metrics.csv")
solver_projection.save_model(path="Projection/kinn.pt")
