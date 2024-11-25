# kkt_framework.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import grad, vmap, jacrev
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import R2Score, MeanAbsolutePercentageError, MeanSquaredError
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import os
import cvxpy as cp

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Variable:
    def __init__(self, name, min_val, max_val):
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def get_min(self, denorm_params):
        if callable(self.min_val):
            return self.min_val(denorm_params)
        else:
            # Se denorm_params è None, restituisce un tensore con il valore costante
            if denorm_params is None or not denorm_params:
                return torch.tensor(self.min_val, dtype=torch.float32)
            # Altrimenti, restituisce un tensore pieno con il valore costante
            return torch.full(
                (denorm_params[next(iter(denorm_params))].shape[0],),
                self.min_val,
                dtype=torch.float32,
                device=denorm_params[next(iter(denorm_params))].device
            )

    def get_max(self, denorm_params):
        if callable(self.max_val):
            return self.max_val(denorm_params)
        else:
            if denorm_params is None or not denorm_params:
                return torch.tensor(self.max_val, dtype=torch.float32)
            return torch.full(
                (denorm_params[next(iter(denorm_params))].shape[0],),
                self.max_val,
                dtype=torch.float32,
                device=denorm_params[next(iter(denorm_params))].device
            )


class Constraint:
    def __init__(self, expr_func, type='inequality'):
        """
        Represents a constraint in the optimization problem.
        
        Args:
            expr_func (callable): Function that takes decision variables and parameters and returns a PyTorch expression.
            type (str): Type of constraint, either 'equality' or 'inequality'.
        """
        self.expr_func = expr_func
        assert type in ['equality', 'inequality'], "Constraint type must be 'equality' or 'inequality'"
        self.type = type
        self.n_constraints = None

    def get_constraints(self, decision_vars, params):
        """
        Computes the constraint expression based on decision variables and parameters.
        
        Args:
            decision_vars (torch.Tensor): Decision variables.
            params (torch.Tensor): Parameters.
        
        Returns:
            torch.Tensor: Constraint expression.
        """
        expr = self.expr_func(decision_vars, params)
        if isinstance(expr, list):
            expr = torch.stack(expr, dim=-1)  # Converti la lista in un tensor
        elif isinstance(expr, torch.Tensor):
            expr = expr.reshape(-1)  # Assumi che la forma sia corretta
        else:
            raise TypeError("expr_func deve restituire un torch.Tensor o una lista di torch.Tensor")

        if self.n_constraints is None:
            # Inferisci il numero di vincoli basandosi sulla dimensione dell'espressione
            self.n_constraints = expr.shape[-1]
        return expr

class OptimizationProblem:
    def __init__(self, parameters, decision_variables, cost_function, constraints):
        """
        Defines an optimization problem.
        
        Args:
            parameters (list of Variable): Parameter variables.
            decision_variables (list of Variable): Decision variables.
            cost_function (callable): Cost function that takes decision variables and parameters.
            constraints (list of Constraint): List of constraints.
        """
        self.parameters = parameters
        self.decision_variables = decision_variables
        self.cost_function = cost_function
        self.constraints = constraints
        self.num_eq_constraints, self.num_ineq_constraints = self.count_constraints()
    def count_constraints(self):
        """
        Conta il numero totale di vincoli di uguaglianza e di disuguaglianza.

        Returns:
            tuple: (num_eq_constraints, num_ineq_constraints)
        """
        num_eq = []
        num_ineq = []
        dummy_batch = torch.zeros(len(self.parameters))  # Batch fittizio per inferire i vincoli

        for constraint in self.constraints:
            expr = constraint.get_constraints(torch.zeros(len(self.decision_variables)), dummy_batch)
            if constraint.type == 'equality':
                num_eq.append(expr.shape[-1])
            elif constraint.type == 'inequality':
                num_ineq.append(expr.shape[-1])
        return num_eq, num_ineq
    def denormalize_parameters(self, norm_params):
        """
        Denormalizza i parametri da [-1, 1] alla loro scala originale sequenzialmente.
        
        Args:
            norm_params (torch.Tensor): Parametri normalizzati di forma (batch_size, num_parameters).
        
        Returns:
            dict: Dizionario dei parametri denormalizzati {nome_parametro: tensor_denormalizzato}.
        """
        denorm_params = {}
        for i, var in enumerate(self.parameters):  # Itera direttamente su self.parameters
            denorm_min = var.get_min(denorm_params)
            denorm_max = var.get_max(denorm_params)
            denorm = 0.5 * (norm_params[:, i] + 1.0) * (denorm_max - denorm_min) + denorm_min
            denorm_params[var.name] = denorm
        return torch.stack([denorm_params[var.name] for var in self.parameters], dim=1), denorm_params
    
    
    def denormalize_decision(self, norm_decisions, denorm_params):
        """
        Denormalizza le variabili decisionali da [-1, 1] alla loro scala originale.
        
        Args:
            norm_decisions (torch.Tensor): Variabili decisionali normalizzate di forma (batch_size, num_decision_vars).
            denorm_params (dict): Dizionario dei parametri denormalizzati {nome_parametro: tensor_denormalizzato}.
        
        Returns:
            torch.Tensor: Variabili decisionali denormalizzate di forma (batch_size, num_decision_vars).
        """
        denorm_decisions = []
        for i, var in enumerate(self.decision_variables):
            denorm_min = var.get_min(denorm_params)
            denorm_max = var.get_max(denorm_params)
            denorm = 0.5 * (norm_decisions[:, i] + 1.0) * (denorm_max - denorm_min) + denorm_min
            denorm_decisions.append(denorm)
        denorm_decisions = torch.stack(denorm_decisions, dim=1)
        return denorm_decisions

class ResidualBlock(nn.Module):
    def __init__(self, n):
        """
        Defines a residual block used in the neural network.
        
        Args:
            n (int): Number of input and output features.
        """
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.bn = nn.BatchNorm1d(n)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        """
        Forward pass of the residual block.
        
        Args:
            X (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        identity = X
        out = self.linear(X)
        out = self.relu(out)
        return self.relu(out + identity)

class KKTNN(nn.Module):
    def __init__(self, input_dim, num_eq_constraints, num_ineq_constraints, num_decision_vars, hidden_dim=512, num_residual_blocks=4):
        """
        Neural network to learn optimal solutions based on KKT conditions.
        
        Args:
            input_dim (int): Number of parameters.
            num_eq_constraints (int): Number of equality constraints.
            num_ineq_constraints (int): Number of inequality constraints.
            num_decision_vars (int): Number of decision variables.
            hidden_dim (int, optional): Hidden layer size. Defaults to 512.
            num_residual_blocks (int, optional): Number of residual blocks. Defaults to 4.
        """
        super(KKTNN, self).__init__()
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        ]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dim))
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.shared = nn.Sequential(*layers)
        
        # Output for decision variables
        self.decision_output = nn.Sequential(
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Linear(hidden_dim, num_decision_vars),  # Outputs between -1 and 1
        )
        
        # Conditionally create outputs for dual variables
        if num_eq_constraints > 0:
            self.dual_eq_output = nn.Sequential(
                ResidualBlock(hidden_dim),
                ResidualBlock(hidden_dim),
                nn.Linear(hidden_dim, num_eq_constraints),
            )
        else:
            self.dual_eq_output = None  # No equality constraints
        
        if num_ineq_constraints > 0:
            self.dual_ineq_output = nn.Sequential(
                ResidualBlock(hidden_dim),
                ResidualBlock(hidden_dim),
                nn.Linear(hidden_dim, num_ineq_constraints),
                nn.Softplus(beta=5)  # Ensures outputs are non-negative
            )
        else:
            self.dual_ineq_output = None  # No inequality constraints

    def forward(self, x):
        """
        Forward pass of the neural network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        
        Returns:
            tuple: Outputs for decision variables, dual equality variables, and dual inequality variables.
        """
        embedding = self.shared(x)
        decision = self.decision_output(embedding)
        dual_eq = self.dual_eq_output(embedding) if self.dual_eq_output is not None else None
        dual_ineq = self.dual_ineq_output(embedding) if self.dual_ineq_output is not None else None
        return decision, dual_eq, dual_ineq

class EarlyStopper:
    def __init__(self, patience=1000, min_delta=1e-4, mode='min'):
        """
        Manages early stopping based on metric improvements.
        
        Args:
            patience (int, optional): Number of steps with no improvement before stopping. Defaults to 1000.
            min_delta (float, optional): Minimum change to qualify as an improvement. Defaults to 1e-4.
            mode (str, optional): 'min' or 'max' based on whether lower or higher values are better. Defaults to 'min'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float('inf') if mode == 'min' else -float('inf')

    def early_stop_triggered(self, current):
        """
        Determines whether early stopping should be triggered.
        
        Args:
            current (float): Current metric value.
        
        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.mode == 'min':
            if current < self.best - self.min_delta:
                self.best = current
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            if current > self.best + self.min_delta:
                self.best = current
                self.counter = 0
            else:
                self.counter += 1
        else:
            raise ValueError("Mode must be 'min' or 'max'")
        
        if self.counter >= self.patience:
            return True
        return False

class ValidationDataset(Dataset):
    def __init__(self, problem, filepath, transform=None):
        """
        Loads a validation dataset from a pickle file.
        
        Args:
            problem (OptimizationProblem): Instance of the optimization problem.
            filepath (str): Path to the pickle file containing the dataset.
            transform (callable, optional): Transformation to apply to the data. Defaults to None.
        """
        self.problem = problem
        self.samples = pickle.load(open(filepath, "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        params, solution = self.samples[idx]
        params = np.array(params, dtype=np.float32)
        solution = np.array(solution, dtype=np.float32)
        if self.transform:
            params = self.transform(params)
            solution = self.transform(solution)
        return torch.tensor(params), torch.tensor(solution)

class KKT_NN:
    def __init__(self, problem, validation_filepath, 
                 learning_rate=3e-4, patience=1000, device=None):
        """
        Initializes the KKT-based neural network model, optimizer, scheduler, early stopper, and logging.
        
        Args:
            problem (OptimizationProblem): Instance of the optimization problem.
            validation_filepath (str): Path to the pickle file containing the validation dataset.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
            patience (int, optional): Number of steps with no improvement before stopping. Defaults to 1000.
            device (str, optional): 'cuda' or 'cpu'. If None, automatically selects based on availability. Defaults to None.
        """
        self.problem = problem
        self.device = device if device else ('cpu' if torch.cuda.is_available() else 'cpu')
        
        # Input dimension: number of parameters
        self.input_dim = len(problem.parameters)
        # Number of equality and inequality constraints
        self.num_eq_constraints = self.problem.num_eq_constraints
        self.num_ineq_constraints = self.problem.num_ineq_constraints
        # Number of decision variables
        self.num_decision_vars = len(problem.decision_variables)
        
        # Initialize the neural network model
        self.model = KKTNN(self.input_dim, sum(self.num_eq_constraints), sum(self.num_ineq_constraints), self.num_decision_vars).to(self.device)
        # Initialize the optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience, factor=0.1, verbose=True)
        # Initialize the early stopper
        self.es = EarlyStopper(patience=patience, min_delta=1e-4, mode='min')  # Based on minimum validation loss
        
        # Initialize TensorBoard logger
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("runs/kkt_nn/" + current_time)
        
        # Initialize metrics dictionary
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'r2': [],
            'mape': [],
            'rmse': []
        }
        
        # Initialize Sobol engine for parameter sampling
        self.sobol_eng = torch.quasirandom.SobolEngine(dimension=self.input_dim, scramble=True, seed=42)
    
    def load_validation_dataset(self, filepath, batch_size=512):
        """
        Loads the validation dataset from a pickle file and returns a DataLoader.
        
        Args:
            filepath (str): Path to the pickle file containing the dataset.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 512.
        
        Returns:
            DataLoader: DataLoader for the validation dataset.
        """
        validation_dataset = ValidationDataset(self.problem, filepath)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        return validation_loader
    
    
    def lagrangian(self, decision_var, dual_eq, dual_ineq, param):
            """
            Calcola il Lagrangiano per un singolo campione.

            Args:
                decision_var (torch.Tensor): Variabile decisionale denormalizzata, forma (num_decision_vars,).
                dual_eq (torch.Tensor or None): Variabili duali per vincoli di uguaglianza, forma (num_eq_constraints,).
                dual_ineq (torch.Tensor or None): Variabili duali per vincoli di disuguaglianza, forma (num_ineq_constraints,).
                param (torch.Tensor): Parametri denormalizzati, forma (num_parameters,).

            Returns:
                torch.Tensor: Valore scalare del Lagrangiano.
            """
            # Calcola il costo
            lagrangian = self.problem.cost_function(decision_var, param)  # Assicurati che cost_function abbia un metodo compute_pytorch
            #lagrangian = cost.squeeze(0)  # Scalar

            # Aggiungi i termini duali per i vincoli di uguaglianza
            if dual_eq is not None:
                expr = torch.cat([constraint.get_constraints(decision_var, param) for constraint in self.problem.constraints[:len(self.num_eq_constraints)]], -1)
                lagrangian += torch.einsum('...i,...i->...', dual_eq, expr)

            # Aggiungi i termini duali per i vincoli di disuguaglianza
            if dual_ineq is not None:
                expr = torch.cat([constraint.get_constraints(decision_var, param) for constraint in self.problem.constraints[len(self.num_eq_constraints):]], -1)
                lagrangian += torch.einsum('...i,...i->...', dual_ineq, expr)

            return lagrangian
    def kkt_loss(self, decision_vars, dual_eq_vars, dual_ineq_vars, parameters):
        """
        Calcola la perdita basata sulle condizioni KKT utilizzando vmap e grad di Functorch.

        Args:
            decision_vars (torch.Tensor): Variabili decisionali denormalizzate, forma (batch_size, num_decision_vars).
            dual_eq_vars (torch.Tensor or None): Variabili duali per i vincoli di uguaglianza, forma (batch_size, num_eq_constraints).
            dual_ineq_vars (torch.Tensor or None): Variabili duali per i vincoli di disuguaglianza, forma (batch_size, num_ineq_constraints).
            parameters (torch.Tensor): Parametri denormalizzati, forma (batch_size, num_parameters).

        Returns:
            tuple: (stationarity_loss, feasibility_loss, complementarity_loss)
        """
        batch_size = decision_vars.shape[0]

        # Se dual_eq_vars è None, sostituiscilo con un tensore di zeri
        if dual_eq_vars is None and sum(self.num_eq_constraints) > 0:
            dual_eq_vars = torch.zeros((batch_size, sum(self.num_eq_constraints)), device=self.device)
        elif self.num_eq_constraints == 0:
            dual_eq_vars = None  # Nessun vincolo di uguaglianza

        # Se dual_ineq_vars è None, sostituiscilo con un tensore di zeri
        if dual_ineq_vars is None and sum(self.num_ineq_constraints) > 0:
            dual_ineq_vars = torch.zeros((batch_size, sum(self.num_ineq_constraints)), device=self.device)
        elif self.num_ineq_constraints == 0:
            dual_ineq_vars = None  # Nessun vincolo di disuguaglianza


        if dual_eq_vars is not None and dual_ineq_vars is not None:
            in_dims = (0, 0, 0, 0)
        elif dual_eq_vars is None and dual_ineq_vars is not None:
            in_dims = (0, None, 0, 0)
        elif dual_eq_vars is not None and dual_ineq_vars is None:
            in_dims = (0, 0, None, 0)
        else:
            in_dims = (0, None, None, 0)

    # Vettorializza la funzione del gradiente del Lagrangiano
        grad_L = vmap(grad(self.lagrangian, argnums=0), in_dims=in_dims)(decision_vars, dual_eq_vars, dual_ineq_vars, parameters)


        # Calcola la perdita di stazionarietà: somma dei quadrati dei gradienti
        loss_stationarity = torch.square(grad_L).sum(1)

        # Calcola la perdita di fattibilità: somma dei quadrati delle violazioni dei vincoli
        feasibility_loss = 0.0
        for i, constraint in enumerate(self.problem.constraints):
            expr = constraint.get_constraints(decision_vars, parameters)  # Forma: (batch_size, n_constraints)
            if constraint.type == 'equality':
                feasibility_loss += torch.square(expr).sum(-1)
            elif constraint.type == 'inequality':
                feasibility_loss += torch.square(torch.relu(expr)).sum(-1)

        # Calcola la perdita di complementarietà: somma dei quadrati del prodotto dual * vincolo
        complementarity_loss = 0.0
        if dual_ineq_vars is not None:
                expr = torch.cat([constraint.get_constraints(decision_vars, parameters) for constraint in self.problem.constraints[len(self.num_eq_constraints):]], -1)
                complementarity = dual_ineq_vars * expr  # Forma: (batch_size,)
                complementarity_loss += torch.square(complementarity).sum(-1)

        return loss_stationarity, feasibility_loss, complementarity_loss

    def training_step(self, batch_size):
        """
        Performs a single training step.
        
        Args:
            batch_size (int): Size of the training batch.
        
        Returns:
            float: Training loss for the current step.
        """
        self.model.train()
        # Sample normalized parameters using Sobol sequence
        norm_params = self.sample_parameters(batch_size)  # Shape: (batch_size, num_parameters)
        # Denormalize parameters to their original scale


        
        params, params_dict = self.problem.denormalize_parameters(norm_params)
        # Forward pass through the model
        decision_vars, dual_eq_vars, dual_ineq_vars = self.model(norm_params)
        
        # Denormalize decision variables
        #decision_vars = self.problem.denormalize_decision(norm_decision_vars, params_dict)
        
        # Compute KKT-based loss
        stationarity_loss, feasibility_loss, complementarity_loss = self.kkt_loss(decision_vars, dual_eq_vars, dual_ineq_vars, params)
        loss = (stationarity_loss + feasibility_loss + complementarity_loss).mean()
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item(), stationarity_loss.mean().item(), feasibility_loss.mean().item(), complementarity_loss.mean().item()

    def validation_step(self, validation_loader):
        """
        Performs a validation step over the entire validation dataset.
        
        Args:
            validation_loader (DataLoader): DataLoader for the validation dataset.
        
        Returns:
            tuple: (average validation loss, average R2 score, average MAPE, average RMSE)
        """
        self.model.eval()
        val_losses = []
        r2_scores = []
        mapes = []
        rmses = []
        with torch.no_grad():
            for params, solutions in validation_loader:
                params = params.to(self.device)
                solutions = solutions.to(self.device)
                
                # Denormalize parameters
                #params_norm, params_dict = self.problem.normalize_parameters(params)
                
                # Forward pass through the model
                decision_vars, dual_eq_vars, dual_ineq_vars = self.model(params)
                
                # Denormalize decision variables
                #decision_vars = self.problem.denormalize_decision(norm_decision_vars, params_dict)
                
                
                
                # Compute metrics
                r2 = R2Score(len(self.problem.decision_variables)).to(self.device)(decision_vars, solutions)
                mape = MeanAbsolutePercentageError().to(self.device)(decision_vars, solutions)
                rmse = MeanSquaredError(squared=False).to(self.device)(decision_vars, solutions)
                
                r2_scores.append(r2.item())
                mapes.append(mape.item())
                rmses.append(rmse.item())
        
        # Calculate average metrics
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mapes)
        avg_rmse = np.mean(rmses)
        
        return avg_r2, avg_mape, avg_rmse

    def sample_parameters(self, batch_size):
        """
        Samples a batch of parameters using the Sobol sequence and normalizes them to [-1, 1].
        
        Args:
            batch_size (int): Number of samples to generate.
        
        Returns:
            torch.Tensor: Normalized parameters of shape (batch_size, num_parameters).
        """
        with torch.no_grad():
            samples = self.sobol_eng.draw(batch_size).to(self.device)
            # Normalize to [-1, 1]
            norm_params = 2.0 * samples - 1.0
            return norm_params

    def train_model(self, num_steps, batch_size, validation_loader, checkpoint_interval=1000):
        """
        Initiates the training process.
        
        Args:
            num_steps (int): Total number of training steps.
            batch_size (int): Size of each training batch.
            validation_loader (DataLoader): DataLoader for the validation dataset.
            checkpoint_interval (int, optional): Interval of steps to save checkpoints. Defaults to 1000.
        """
        for step in tqdm(range(1, num_steps + 1)):
            # Perform a training step
            train_loss, stationarity_loss, feasibility_loss, complementarity_loss = self.training_step(batch_size)
            self.metrics['train_loss'].append(train_loss)
            self.tb_logger.add_scalar("Train/Loss", train_loss, step)
            
            # Perform validation periodically
            if step % 1 == 0 or step == num_steps:
                r2, mape, rmse = self.validation_step(validation_loader)
                self.metrics['r2'].append(r2)
                self.metrics['mape'].append(mape)
                self.metrics['rmse'].append(rmse)
                
                self.tb_logger.add_scalar("Val/R2", r2, step)
                self.tb_logger.add_scalar("Val/MAPE", mape, step)
                self.tb_logger.add_scalar("Val/RMSE", rmse, step)
                
                print(f"Step {step}: Train Loss={train_loss:.6f}, Stationarity Loss={stationarity_loss:.6f}, Feasibility Loss={feasibility_loss:.6f}, Complementarity Loss={complementarity_loss:.6f}, Val R2={r2:.4f}, MAPE={mape:.4f}, RMSE={rmse:.6f}")
                
                # Update the learning rate scheduler
                self.scheduler.step(train_loss)
                
                # Check for early stopping
                if self.es.early_stop_triggered(train_loss):
                    print("Early stopping triggered")
                    break
            
            # Save checkpoints at specified intervals
            if step % checkpoint_interval == 0:
                self.save_checkpoint(step)
                print(f"Checkpoint saved at step {step}.")

    def save_checkpoint(self, step, filepath=None):
        """
        Saves a checkpoint of the current model state.
        
        Args:
            step (int): Current training step.
            filepath (str, optional): Path to save the checkpoint. If None, uses a default naming convention. Defaults to None.
        """
        if filepath is None:
            filepath = f"checkpoint_step_{step}.pth"
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': self.metrics,
            'es_state': {
                'counter': self.es.counter,
                'best': self.es.best
            }
        }
        torch.save(checkpoint, filepath)

    def save_model(self, filepath):
        """
        Saves the current model state.
        
        Args:
            filepath (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """
        Loads a saved model state.
        
        Args:
            filepath (str): Path from which to load the model.
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.model.to(self.device)

    def save_metrics(self, filepath):
        """
        Saves the training and validation metrics to a CSV file.
        
        Args:
            filepath (str): Path to save the metrics CSV.
        """
        pd.DataFrame(self.metrics).to_csv(filepath, index=False)

    def load_metrics(self, filepath):
        """
        Loads training and validation metrics from a CSV file.
        
        Args:
            filepath (str): Path from which to load the metrics CSV.
        """
        df = pd.read_csv(filepath)
        for column in df.columns:
            self.metrics[column] = df[column].tolist()

    def predict(self, input_params):
        """
        Computes the optimal solution given unnormalized input parameters.
        
        Args:
            input_params (list or np.ndarray or torch.Tensor): Unnormalized input parameters.
        
        Returns:
            np.ndarray: Optimal solution.
        """
        self.model.eval()
        
        # Convert input_params to Tensor if necessary
        if isinstance(input_params, np.ndarray):
            input_params = torch.from_numpy(input_params).float()
        elif isinstance(input_params, list):
            input_params = torch.tensor(input_params, dtype=torch.float32)
        elif not isinstance(input_params, torch.Tensor):
            raise TypeError("input_params must be array-like or a Tensor.")
        
        # Add batch dimension if necessary
        if input_params.dim() == 1:
            input_params = input_params.unsqueeze(0)  # Shape: (1, num_parameters)
        
        input_params = input_params.to(self.device)
        
        with torch.no_grad():
            # Normalize parameters to [-1, 1]
            min_vals = torch.tensor([var.min_val for var in self.problem.parameters], dtype=torch.float32).to(self.device)
            max_vals = torch.tensor([var.max_val for var in self.problem.parameters], dtype=torch.float32).to(self.device)
            norm_params = 2.0 * (input_params - min_vals) / (max_vals - min_vals) - 1.0
            
            # Forward pass through the model
            norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(norm_params)
            
            # Denormalize decision variables
            decision_vars = self.problem.denormalize_decision(norm_decision_vars, input_params)
        
        # Remove batch dimension if it was added
        if decision_vars.size(0) == 1:
            decision_vars = decision_vars.squeeze(0)
        
        return decision_vars.cpu().numpy()

    def load_checkpoint(self, filepath):
        """
        Loads a model checkpoint.
        
        Args:
            filepath (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.metrics = checkpoint['metrics']
        self.es.counter = checkpoint['es_state']['counter']
        self.es.best = checkpoint['es_state']['best']
        self.model.to(self.device)