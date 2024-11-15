# src/solver.py

import torch
import random
import numpy as np
import pandas as pd
import pickle
from torch import nn, optim
from torch.func import grad, vmap
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchmetrics import R2Score, MeanAbsolutePercentageError, MeanSquaredError
from datetime import datetime
from tqdm import tqdm
from typing import Callable, List, Optional, Dict, Tuple

# Impostazioni di riproducibilità
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

set_seed()

# Classi per i dataset
class ConvexOptimizationDataset(Dataset):
    def __init__(self, data_path: str, normalization: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None):
        """
        Carica il dataset da un file pickle.

        Parametri:
        - data_path: Percorso al file del dataset (pickle).
        - normalization: Dizionario che mappa l'indice della variabile all'intervallo di normalizzazione.
        """
        with open(data_path, "rb") as f:
            self.samples = pickle.load(f)
        self.normalization = normalization if normalization else {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X, y = self.samples[idx]
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        # Applica normalizzazione se definita
        if self.normalization:
            for key, func in self.normalization.items():
                if isinstance(key, int):
                    X[key] = func(X[key])
                elif isinstance(key, slice):
                    X[key] = func(X[key])
                # Estendere se necessario
        return torch.tensor(X), torch.tensor(y)

class SobolConvexOptimizationDataset(Dataset):
    def __init__(self, num_samples: int, input_dim: int, normalization: Optional[Dict[int, Callable[[np.ndarray], np.ndarray]]] = None):
        """
        Genera il dataset utilizzando sequenze di Sobol.

        Parametri:
        - num_samples: Numero totale di campioni da generare.
        - input_dim: Dimensione dell'input.
        - normalization: Dizionario che mappa l'indice della variabile all'intervallo di normalizzazione.
        """
        from torch.quasirandom import SobolEngine  # Import locale per evitare problemi di dipendenze
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.normalization = normalization if normalization else {}
        self.engine = SobolEngine(dimension=input_dim, scramble=True)
        # Genera tutti i campioni una volta per efficienza
        self.samples = self.engine.draw(n=self.num_samples).numpy()  # [num_samples, input_dim]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.samples[idx]
        y = X[:2]  # Supponiamo che le prime 2 colonne siano le azioni

        # Applica normalizzazione se definita
        if self.normalization:
            for key, func in self.normalization.items():
                if isinstance(key, int):
                    X[key] = func(X[key])
                elif isinstance(key, slice):
                    X[key] = func(X[key])
                # Estendere se necessario

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Classi di supporto
class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class ResidualBlock(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        identity = X
        out = self.linear(X)
        out = self.relu(out + identity)
        out = self.ln(out)
        out = self.dropout(out)
        return out

class NeuralNetwork(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        shared_layers: List[int], 
        sol_layers: List[int], 
        lambda_layers: List[int], 
        output_dim_sol: int, 
        output_dim_lambda: int, 
        activation_sol: Optional[nn.Module] = None
    ):
        super().__init__()
        layers = []
        for layer_size in shared_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.LeakyReLU())
            layers.append(ResidualBlock(layer_size))
            input_dim = layer_size
        self.shared = nn.Sequential(*layers)

        sol_layers_full = []
        for layer_size in sol_layers:
            sol_layers_full.append(ResidualBlock(layer_size))
        sol_layers_full.append(nn.Linear(sol_layers[-1], output_dim_sol))
        if activation_sol:
            sol_layers_full.append(activation_sol)
        self.sol = nn.Sequential(*sol_layers_full)

        lambda_layers_full = []
        for layer_size in lambda_layers:
            lambda_layers_full.append(ResidualBlock(layer_size))
        lambda_layers_full.append(nn.Linear(lambda_layers[-1], output_dim_lambda))
        self.lambda_net = nn.Sequential(*lambda_layers_full, nn.Softplus(beta=5))  # Softplus per garantire positività

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.shared(X)
        sol = self.sol(embedding)
        lambda_ = self.lambda_net(embedding)
        return sol, lambda_

class ConvexOptimizationSolver:
    def __init__(
        self,
        input_dim: int,
        shared_layers: List[int],
        sol_layers: List[int],
        lambda_layers: List[int],
        output_dim_sol: int,
        output_dim_lambda: int,
        activation_sol: Optional[nn.Module],
        cost_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        equality_constraints: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        inequality_constraints: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        action_indices: Optional[List[int]] = None,
        variable_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        dataset_path: Optional[str] = None,  # Per la validazione
        batch_size: int = 512,
        learning_rate: float = 3e-4,
        patience: int = 1000,
        optimizer_type: str = 'adam',
        scheduler_type: Optional[str] = 'reduce_on_plateau',
        device: Optional[str] = None,
        num_samples: int = 100000  # Numero di campioni da generare tramite Sobol per l'addestramento
    ):
        """
        Inizializza il solver per problemi di ottimizzazione convessa.

        Parametri:
        - input_dim: Dimensione dell'input.
        - shared_layers: Lista di dimensioni per i layer condivisi.
        - sol_layers: Lista di dimensioni per i layer della soluzione.
        - lambda_layers: Lista di dimensioni per i layer dei moltiplicatori di Lagrange.
        - output_dim_sol: Dimensione dell'output della soluzione.
        - output_dim_lambda: Dimensione dell'output dei moltiplicatori di Lagrange.
        - activation_sol: Attivazione finale per la soluzione (es. nn.Sigmoid(), nn.Tanh()).
        - cost_function: Funzione di costo personalizzata.
        - equality_constraints: Funzione di vincolo di uguaglianza che restituisce un tensore [batch_size, num_eq_constraints].
        - inequality_constraints: Funzione di vincolo di disuguaglianza che restituisce un tensore [batch_size, num_ineq_constraints].
        - action_indices: Lista degli indici delle variabili di input che rappresentano le azioni.
        - variable_ranges: Dizionario che mappa l'indice della variabile all'intervallo (min, max) per la normalizzazione.
        - dataset_path: Percorso al file del dataset (pickle) per la validazione.
        - batch_size: Dimensione del batch.
        - learning_rate: Tasso di apprendimento.
        - patience: Numero di iterazioni senza miglioramento prima dello stop.
        - optimizer_type: Tipo di ottimizzatore ('adam', 'lbfgs', etc.).
        - scheduler_type: Tipo di scheduler ('reduce_on_plateau', 'exponential', etc.).
        - device: Dispositivo di calcolo ('cpu', 'cuda', 'mps').
        - num_samples: Numero di campioni da generare tramite Sobol per l'addestramento.
        """
        self.device = device if device else ('cpu' if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        self.output_dim_sol = output_dim_sol
        self.output_dim_lambda = output_dim_lambda
        self.cost_function = cost_function
        self.equality_constraints = equality_constraints  # Funzione che restituisce [batch_size, num_eq_constraints]
        self.inequality_constraints = inequality_constraints  # Funzione che restituisce [batch_size, num_ineq_constraints]
        self.action_indices = action_indices if action_indices else list(range(input_dim))  # Default: tutte le inputs sono azioni
        self.variable_ranges = variable_ranges if variable_ranges else {}

        # Creazione delle funzioni di normalizzazione e unnormalizzazione
        self.normalization_funcs = {}
        self.unnormalization_funcs = {}
        for idx, (min_val, max_val) in self.variable_ranges.items():
            self.normalization_funcs[idx] = lambda x, min_val=min_val, max_val=max_val: 2.0 * (x - min_val) / (max_val - min_val) - 1.0
            self.unnormalization_funcs[idx] = lambda x, min_val=min_val, max_val=max_val: 0.5 * (x + 1.0) * (max_val - min_val) + min_val

        # Inizializzazione della rete neurale
        self.model = NeuralNetwork(
            input_dim=input_dim,
            shared_layers=shared_layers,
            sol_layers=sol_layers,
            lambda_layers=lambda_layers,
            output_dim_sol=output_dim_sol,
            output_dim_lambda=output_dim_lambda,
            activation_sol=activation_sol
        ).to(self.device)

        # Ottimizzatore
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == 'lbfgs':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Ottimizzatore {optimizer_type} non supportato.")

        # Scheduler
        if scheduler_type:
            if scheduler_type.lower() == 'reduce_on_plateau':
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.1)
            elif scheduler_type.lower() == 'exponential':
                self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
            else:
                self.scheduler = None
        else:
            self.scheduler = None

        # Early stopper
        self.early_stopper = EarlyStopper(patience=patience)
        self.terminated = False
        self.n_iter = 0

        # Logger
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter(f"runs/tb_logs/{current_time}")

        # Logging
        self.loss_stationarity = []
        self.loss_feasibility = []
        self.loss_complementarity = []
        self.metrics = {'R2': [], 'MAPE': [], 'MAE': [], 'RMSE': []}

        # Dataset e DataLoader
        if num_samples and dataset_path:
            # Inizializza entrambi i dataset
            self.train_dataset = SobolConvexOptimizationDataset(
                num_samples=num_samples,
                input_dim=input_dim,
                normalization=self.normalization_funcs
            )
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

            self.val_dataset = ConvexOptimizationDataset(
                data_path=dataset_path,
                normalization=self.normalization_funcs
            )
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        else:
            raise ValueError("Devi fornire sia un numero di campioni per l'addestramento che un percorso al dataset per la validazione.")

    def set_constraints(self, equality_constraints: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                       inequality_constraints: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None):
        if equality_constraints:
            self.equality_constraints = equality_constraints
        if inequality_constraints:
            self.inequality_constraints = inequality_constraints

    def set_cost_function(self, cost_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.cost_function = cost_fn

    def compute_kkt_loss(self, inputs: torch.Tensor, sol: torch.Tensor, lambda_: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calcola la perdita basata sulle condizioni KKT.
        """
        # Stazionarietà: ∇_x L = 0
        lagrangian_grad = vmap(grad(self.lagrangian, argnums=1))(inputs, sol, lambda_)
        loss_stationarity = torch.square(lagrangian_grad).sum(dim=-1)

        # Fattibilità
        feasibility_eq = self.equality_constraints(inputs, sol) if self.equality_constraints else torch.tensor(0.0, device=self.device)
        feasibility_ineq = self.inequality_constraints(inputs, sol) if self.inequality_constraints else torch.tensor(0.0, device=self.device)
        
        loss_feasibility_eq = torch.square(feasibility_eq).sum(dim=-1) if self.equality_constraints else torch.tensor(0.0, device=self.device)
        loss_feasibility_ineq = torch.square(torch.relu(feasibility_ineq)).sum(dim=-1) if self.inequality_constraints else torch.tensor(0.0, device=self.device)
        loss_feasibility = loss_feasibility_eq + loss_feasibility_ineq

        # Complementarità: lambda * g(x) = 0 (solo per vincoli di disuguaglianza)
        complementarity = lambda_ * torch.relu(feasibility_ineq) if self.inequality_constraints else torch.tensor(0.0, device=self.device)
        loss_complementarity = torch.square(complementarity).sum(dim=-1) if self.inequality_constraints else torch.tensor(0.0, device=self.device)

        return loss_stationarity, loss_feasibility, loss_complementarity

    def lagrangian(self, inputs: torch.Tensor, sol: torch.Tensor, lambda_: torch.Tensor) -> torch.Tensor:
        """
        Calcola il Lagrangiano.
        """
        if not self.cost_function:
            raise ValueError("Devi definire una funzione di costo.")
        if not self.equality_constraints and not self.inequality_constraints:
            raise ValueError("Devi definire almeno una funzione di vincolo.")

        # Inizializza il Lagrangiano con la funzione di costo
        L = self.cost_function(inputs, sol)

        # Combina vincoli di uguaglianza e disuguaglianza
        total_constraints = []
        if self.equality_constraints:
            total_constraints.append(self.equality_constraints(inputs, sol))
        if self.inequality_constraints:
            total_constraints.append(self.inequality_constraints(inputs, sol))

        if total_constraints:
            constraints = torch.cat(total_constraints, dim=-1)  # [batch_size, num_constraints]
            L += (lambda_ * constraints).sum(dim=-1)  # [batch_size]

        return L

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        Esegue un passo di addestramento.
        """
        self.model.train()
        actions, targets = batch  # Assicurati che il dataset ritorni le azioni e i target corretti
        actions = actions.to(self.device)
        targets = targets.to(self.device)

        def closure():
            if torch.is_grad_enabled():
                self.optimizer.zero_grad()
            sol, lambda_ = self.model(actions)
            loss_stationarity, loss_feasibility, loss_complementarity = self.compute_kkt_loss(actions, sol, lambda_)
            loss = (loss_stationarity + loss_feasibility + loss_complementarity).mean()
            if torch.is_grad_enabled():
                loss.backward()
                self.optimizer.step()
            # Logging
            self.tb_logger.add_scalar("Loss/Sum", loss.item(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Stationarity", loss_stationarity.mean().item(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Feasibility", loss_feasibility.mean().item(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Complementarity", loss_complementarity.mean().item(), self.n_iter)

            self.loss_stationarity.append(loss_stationarity.mean().item())
            self.loss_feasibility.append(loss_feasibility.mean().item())
            self.loss_complementarity.append(loss_complementarity.mean().item())

            # Early stopping check
            if self.early_stopper.early_stop(loss.item()):
                print("Early stopping triggered.")
                self.terminated = True

            self.n_iter += 1
            return loss

        if isinstance(self.optimizer, optim.LBFGS):
            loss = self.optimizer.step(closure)
        else:
            loss = closure()

        return loss.item()

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """
        Esegue un passo di validazione.
        """
        self.model.eval()
        with torch.no_grad():
            actions, targets = batch
            actions = actions.to(self.device)
            targets = targets.to(self.device)

            sol, lambda_ = self.model(actions)
            # Debug
            # print(f"sol shape: {sol.shape}, targets shape: {targets.shape}")
            val_loss = nn.functional.l1_loss(sol, targets)

            # Metriche
            if self.output_dim_sol == 1:
                r2 = R2Score().to(self.device)(sol, targets).item()
            else:
                r2 = R2Score(num_outputs=self.output_dim_sol).to(self.device)(sol, targets).item()
            mape = MeanAbsolutePercentageError().to(self.device)(sol, targets).item()
            rmse = MeanSquaredError(squared=False).to(self.device)(sol, targets).item()
            mae = nn.functional.l1_loss(sol, targets).item()

            # Logging
            self.metrics['R2'].append(r2)
            self.metrics['MAPE'].append(mape)
            self.metrics['RMSE'].append(rmse)
            self.metrics['MAE'].append(mae)

            # Utilizziamo l'indice dell'epoca per le metriche
            current_epoch = len(self.metrics['R2'])
            self.tb_logger.add_scalar("Val/Loss", val_loss.item(), current_epoch)
            self.tb_logger.add_scalar("Val/R2", r2, current_epoch)
            self.tb_logger.add_scalar("Val/MAPE", mape, current_epoch)
            self.tb_logger.add_scalar("Val/RMSE", rmse, current_epoch)
            self.tb_logger.add_scalar("Val/MAE", mae, current_epoch)

    def train_model(self, epochs: int = 100):
        """
        Esegue l'addestramento del modello per un numero specificato di epoche.
        """
        best_r2 = -float('inf')
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # Fase di addestramento
            for batch in self.train_loader:
                loss = self.training_step(batch)
                if self.terminated:
                    break
            if self.terminated:
                print(f"Terminazione anticipata all'epoca {epoch+1}")
                break
            pbar.set_description(f"Epoch {epoch+1}, Loss: {self.loss_stationarity[-1]:.4f}")

            # Fase di validazione
            for batch in self.val_loader:
                self.validation_step(batch)
                break  # Valida su un batch per velocità

            # Salva il modello se R2 migliora
            current_r2 = self.metrics['R2'][-1]
            if current_r2 > best_r2:
                best_r2 = current_r2
                self.save_model(path="Projection/best_model.pt")
                print(f"Miglior modello salvato all'epoca {epoch+1} con R2: {current_r2:.4f}")

            # Scheduler step (se applicabile)
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(self.loss_feasibility[-1])  # Puoi scegliere quale perdita usare
                else:
                    self.scheduler.step()

        self.tb_logger.close()

    def save_logs(self, log_path: str = "log.csv", metrics_path: str = "metrics.csv"):
        """
        Salva i log delle perdite e delle metriche in file CSV separati.
        """
        # Log delle perdite per ogni batch
        log_df = pd.DataFrame({
            'Step': list(range(1, self.n_iter + 1)),
            'Stationarity': self.loss_stationarity,
            'Feasibility': self.loss_feasibility,
            'Complementarity': self.loss_complementarity
        })
        
        # Log delle metriche per ogni epoca
        metrics_df = pd.DataFrame({
            'Epoch': list(range(1, len(self.metrics['R2']) + 1)),
            'R2': self.metrics['R2'],
            'MAPE': self.metrics['MAPE'],
            'MAE': self.metrics['MAE'],
            'RMSE': self.metrics['RMSE']
        })
        
        # Salva i DataFrame nei rispettivi file CSV
        log_df.to_csv(log_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

    def save_model(self, path: str = "model.pt"):
        """
        Salva lo stato del modello.
        """
        torch.save(self.model.state_dict(), path)
