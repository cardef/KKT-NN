# kkt_framework.py

import torch
import torch.nn as nn
import torch.optim as optim
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
        """
        Represents a parameter or decision variable with static or dynamic bounds.
        
        Args:
            name (str): Name of the variable.
            min_val (float or callable): Minimum value or a function to compute the minimum based on parameters.
            max_val (float or callable): Maximum value or a function to compute the maximum based on parameters.
        """
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def get_min(self, parameters):
        """
        Retrieves the minimum value of the variable.
        
        Args:
            parameters (torch.Tensor): Current parameter values.
        
        Returns:
            torch.Tensor: Tensor containing the minimum values.
        """
        if callable(self.min_val):
            return self.min_val(parameters)
        else:
            return torch.full((parameters.shape[0],), self.min_val, dtype=torch.float32, device=parameters.device)

    def get_max(self, parameters):
        """
        Retrieves the maximum value of the variable.
        
        Args:
            parameters (torch.Tensor): Current parameter values.
        
        Returns:
            torch.Tensor: Tensor containing the maximum values.
        """
        if callable(self.max_val):
            return self.max_val(parameters)
        else:
            return torch.full((parameters.shape[0],), self.max_val, dtype=torch.float32, device=parameters.device)

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

    def get_constraint(self, decision_vars, params):
        """
        Computes the constraint expression based on decision variables and parameters.
        
        Args:
            decision_vars (torch.Tensor): Decision variables.
            params (torch.Tensor): Parameters.
        
        Returns:
            torch.Tensor: Constraint expression.
        """
        expr = self.expr_func(decision_vars, params)
        return expr
        """ if self.type == 'equality':
            return expr  # Must be equal to zero
        elif self.type == 'inequality':
            return torch.relu(expr)  # For constraints <= 0, apply ReLU to ensure non-negativity """

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

    def denormalize_parameters(self, norm_params):
        """
        Denormalizes parameters from [-1, 1] to their original scale.
        
        Args:
            norm_params (torch.Tensor): Normalized parameters of shape (batch_size, num_parameters).
        
        Returns:
            torch.Tensor: Denormalized parameters of shape (batch_size, num_parameters).
        """
        denorm_params = []
        for i, var in enumerate(self.parameters):
            min_val = var.get_min(norm_params)
            max_val = var.get_max(norm_params)
            denorm = 0.5 * (norm_params[:, i] + 1.0) * (max_val - min_val) + min_val
            denorm_params.append(denorm)
        denorm_params = torch.stack(denorm_params, dim=1)
        return denorm_params

    def denormalize_decision(self, norm_decisions, parameters):
        """
        Denormalizes decision variables from [-1, 1] to their original scale.
        
        Args:
            norm_decisions (torch.Tensor): Normalized decision variables of shape (batch_size, num_decision_vars).
            parameters (torch.Tensor): Denormalized parameters of shape (batch_size, num_parameters).
        
        Returns:
            torch.Tensor: Denormalized decision variables of shape (batch_size, num_decision_vars).
        """
        denorm_decisions = []
        for i, var in enumerate(self.decision_variables):
            min_val = var.get_min(parameters)
            max_val = var.get_max(parameters)
            denorm = 0.5 * (norm_decisions[:, i] + 1.0) * (max_val - min_val) + min_val
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
            nn.Linear(hidden_dim, num_decision_vars),
            nn.Tanh()  # Outputs between -1 and 1
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
                nn.Softplus()  # Ensures outputs are non-negative
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

def kkt_loss(problem, decision_vars, dual_eq_vars, dual_ineq_vars, parameters):
    """
    Computes the loss based on KKT conditions.
    
    Args:
        problem (OptimizationProblem): Instance of the optimization problem.
        decision_vars (torch.Tensor): Decision variables.
        dual_eq_vars (torch.Tensor): Dual variables for equality constraints.
        dual_ineq_vars (torch.Tensor): Dual variables for inequality constraints.
        parameters (torch.Tensor): Parameters of the problem.
    
    Returns:
        tuple: (stationarity_loss, feasibility_loss, complementarity_loss)
    """
    # Enable gradient computation for decision variables
    cost = problem.cost_function(decision_vars, parameters)
    
    lagrangian = cost
    if dual_eq_vars is not None:
        for dual_eq, constraint in zip(dual_eq_vars.T, problem.constraints):
            if constraint.type == 'equality':
                lagrangian += dual_eq * constraint.get_constraint(decision_vars, parameters)
    
    if dual_ineq_vars is not None:
        for dual_ineq, constraint in zip(dual_ineq_vars.T, problem.constraints):
            if constraint.type == 'inequality':
                lagrangian += dual_ineq * constraint.get_constraint(decision_vars, parameters)
    
    grad_L = torch.autograd.grad(lagrangian.sum(), decision_vars, retain_graph=True, create_graph=True)[0]
    
    stationarity_loss = torch.mean(torch.sum(grad_L**2, dim=1))
    
    feasibility_loss = 0.0
    for constraint in problem.constraints:
        expr = constraint.get_constraint(decision_vars, parameters)
        if constraint.type == 'equality':
            feasibility_loss += torch.mean(expr**2)
        elif constraint.type == 'inequality':
            feasibility_loss += torch.mean(torch.relu(expr)**2)
    
    complementarity_loss = 0.0
    if dual_ineq_vars is not None:
        for dual_ineq, constraint in zip(dual_ineq_vars.T, problem.constraints):
            if constraint.type == 'inequality':
                expr = constraint.get_constraint(decision_vars, parameters)
                complementarity = dual_ineq * torch.relu(expr)
                complementarity_loss += torch.mean(complementarity**2)
    
    return stationarity_loss, feasibility_loss, complementarity_loss


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
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input dimension: number of parameters
        self.input_dim = len(problem.parameters)
        # Number of equality and inequality constraints
        self.num_eq_constraints = sum(1 for c in problem.constraints if c.type == 'equality')
        self.num_ineq_constraints = sum(1 for c in problem.constraints if c.type == 'inequality')
        # Number of decision variables
        self.num_decision_vars = len(problem.decision_variables)
        
        # Initialize the neural network model
        self.model = KKTNN(self.input_dim, self.num_eq_constraints, self.num_ineq_constraints, self.num_decision_vars).to(self.device)
        # Initialize the optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=patience//10, factor=0.1, verbose=True)
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
    
    def generate_validation_dataset(self, num_samples=1000, solver=cp.ECOS):
        """
        Generates a validation dataset using CVXPY and saves it to a pickle file.
        
        Args:
            num_samples (int, optional): Number of samples to generate. Defaults to 1000.
            solver (cvxpy Solver, optional): Solver to use with CVXPY. Defaults to cp.ECOS.
        """
        generate_validation_dataset_cvxpy(self.problem, num_samples=num_samples, solver=solver)
    
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
        params = self.problem.denormalize_parameters(norm_params)
        
        # Forward pass through the model
        norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(norm_params)
        
        # Denormalize decision variables
        decision_vars = self.problem.denormalize_decision(norm_decision_vars, params)
        
        # Compute KKT-based loss
        stationarity_loss, feasibility_loss, complementarity_loss = kkt_loss(
            self.problem, decision_vars, dual_eq_vars, dual_ineq_vars, params
        )
        loss = stationarity_loss + feasibility_loss + complementarity_loss
        
        # Backpropagation with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

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
            for params_norm, solutions in validation_loader:
                params_norm = params_norm.to(self.device)
                solutions = solutions.to(self.device)
                
                # Denormalize parameters
                params = self.problem.denormalize_parameters(params_norm)
                
                # Forward pass through the model
                norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(params_norm)
                
                # Denormalize decision variables
                decision_vars = self.problem.denormalize_decision(norm_decision_vars, params)
                
                
                
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
            train_loss = self.training_step(batch_size)
            self.metrics['train_loss'].append(train_loss)
            self.tb_logger.add_scalar("Train/Loss", train_loss, step)
            
            # Perform validation periodically
            if step % 100 == 0 or step == num_steps:
                r2, mape, rmse = self.validation_step(validation_loader)
                self.metrics['r2'].append(r2)
                self.metrics['mape'].append(mape)
                self.metrics['rmse'].append(rmse)
                
                self.tb_logger.add_scalar("Val/R2", r2, step)
                self.tb_logger.add_scalar("Val/MAPE", mape, step)
                self.tb_logger.add_scalar("Val/RMSE", rmse, step)
                
                print(f"Step {step}: Train Loss={train_loss:.6f}, R2={r2:.4f}, MAPE={mape:.4f}, RMSE={rmse:.6f}")
                
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

def generate_validation_dataset_cvxpy(problem, num_samples=1000, solver=cp.ECOS):
    """
    Generates a validation dataset using CVXPY and saves it to a pickle file.
    
    Args:
        problem (OptimizationProblem): Instance of the optimization problem.
        num_samples (int, optional): Number of samples to generate. Defaults to 1000.
        solver (cvxpy Solver, optional): Solver to use with CVXPY. Defaults to cp.ECOS.
    """
    sobol_eng = torch.quasirandom.SobolEngine(dimension=len(problem.parameters), scramble=True, seed=42)
    norm_params = 2.0 * sobol_eng.draw(num_samples) - 1.0  # Normalize to [-1, 1]
    params = problem.denormalize_parameters(norm_params).cpu().numpy()

    dataset = []
    for i in tqdm(range(num_samples), desc="Generating validation dataset"):
        param = params[i]
        
        # Define decision variables as CVXPY Variables
        num_decisions = len(problem.decision_variables)
        decision_vars = cp.Variable(num_decisions)
        
        # Define the cost function using CVXPY
        cost_expr = problem.cost_function(decision_vars, param)
        
        # Define constraints using CVXPY expressions directly
        constraints_cvxpy = []
        for constraint in problem.constraints:
            if constraint.type == 'equality':
                constraints_cvxpy.append(constraint.expr_func(decision_vars, param) == 0)
            elif constraint.type == 'inequality':
                constraints_cvxpy.append(constraint.expr_func(decision_vars, param) <= 0)
        
        # Define the optimization problem
        prob = cp.Problem(cp.Minimize(cost_expr), constraints_cvxpy)
        
        # Solve the problem
        try:
            prob.solve(solver=solver)
            if prob.status not in ["infeasible", "unbounded"]:
                solution = decision_vars.value
            else:
                # If infeasible or unbounded, use a random solution within bounds
                solution = np.random.uniform(
                    low=[var.min_val if not callable(var.min_val) else var.get_min(torch.tensor(param, dtype=torch.float32).unsqueeze(0)).item() for var in problem.decision_variables],
                    high=[var.max_val if not callable(var.max_val) else var.get_max(torch.tensor(param, dtype=torch.float32).unsqueeze(0)).item() for var in problem.decision_variables]
                )
        except Exception as e:
            print(f"Error solving problem with parameters {param}: {e}")
            # In case of error, use a random solution
            solution = np.random.uniform(
                low=[var.min_val if not callable(var.min_val) else var.get_min(torch.tensor(param, dtype=torch.float32).unsqueeze(0)).item() for var in problem.decision_variables],
                high=[var.max_val if not callable(var.max_val) else var.get_max(torch.tensor(param, dtype=torch.float32).unsqueeze(0)).item() for var in problem.decision_variables]
            )
        
        # Denormalize the solutions
        solution_tensor = torch.tensor(solution, dtype=torch.float32)
        denorm_solution = problem.denormalize_decision(solution_tensor.unsqueeze(0), torch.tensor(param, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        
        # Append the normalized parameters and denormalized solutions to the dataset
        dataset.append((norm_params[i].cpu().numpy(), denorm_solution))
    
    # Save the dataset to a pickle file
    with open("validation_dataset_cvxpy.pkl", "wb") as f:
        pickle.dump(dataset, f)
