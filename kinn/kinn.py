import torch
import torch.nn as nn
import torch.optim as optim
from torch.func import grad, vmap
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import R2Score, MeanAbsolutePercentageError, MeanSquaredError, MeanAbsoluteError
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
import os
from kinn.model import Net
from torchjd import backward
from torchjd.aggregation import UPGrad
class ValidationDataset(Dataset):
    """
    Dataset class for loading validation data from a pickle file.
    """

    def __init__(self, problem, filepath, transform=None):
        """
        Initializes the ValidationDataset.

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


class EarlyStopper:
    """
    Manages early stopping based on metric improvements.
    """

    def __init__(self, patience=1000, min_delta=0.0, mode="min"):
        """
        Initializes the EarlyStopper.

        Args:
            patience (int, optional): Number of steps with no improvement before stopping. Defaults to 1000.
            min_delta (float, optional): Minimum change to qualify as an improvement. Defaults to 0.0.
            mode (str, optional): 'min' or 'max' based on whether lower or higher values are better. Defaults to 'min'.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = float("inf") if mode == "min" else -float("inf")

    def early_stop_triggered(self, current):
        """
        Determines whether early stopping should be triggered.

        Args:
            current (float): Current metric value.

        Returns:
            bool: True if early stopping is triggered, False otherwise.
        """
        if self.mode == "min":
            if current < self.best - self.min_delta:
                self.best = current
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == "max":
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


class KINN:
    """
    Main class that handles training the KKT-based neural network.
    """

    def __init__(
        self,
        problem,
        validation_filepath = None,
        hidden_dim=512,
        num_embedding_residual_block=4,
        num_outputs_residual_block=4,
        learning_rate=3e-4,
        early_stop_patience=1000,
        early_stop_delta=0.0,
        scheduler_patience=100,
        device=None,
    ):
        """
        Initializes the KKT_NN model, optimizer, scheduler, early stopper, and logging.

        Args:
            problem (OptimizationProblem): Instance of the optimization problem.
            validation_filepath (str, optional): Path to the pickle file containing the validation dataset.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 3e-4.
            early_stop_patience (int, optional): Number of steps with no improvement before stopping. Defaults to 1000.
            early_stop_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.0.
            scheduler_patience (int, optional): Number of steps with no improvement before decreasing learning rate. Defaults to 100.
            device (str, optional): 'cuda' or 'cpu'. If None, automatically selects based on availability. Defaults to None.
        """
        self.problem = problem
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Input dimension: number of parameters
        self.input_dim = len(problem.parameters)
        # Number of equality and inequality constraints
        self.num_eq_constraints = problem.num_eq_constraints  # Now an integer
        self.num_ineq_constraints = problem.num_ineq_constraints  # Now an integer
        # Number of decision variables
        self.num_decision_vars = len(problem.decision_variables)

        # Initialize the neural network model
        self.model = Net(
            input_dim=self.input_dim,
            num_eq_constraints=self.num_eq_constraints,
            num_ineq_constraints=self.num_ineq_constraints,
            num_decision_vars=self.num_decision_vars,
            hidden_dim=hidden_dim,
            num_embedding_residual_blocks=num_embedding_residual_block,
            num_outputs_residual_blocks=num_outputs_residual_block,
        ).to(self.device)
        # Initialize the optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # Initialize the learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=scheduler_patience, factor=0.1
        )
        # Initialize the early stopper
        self.es = EarlyStopper(
            patience=early_stop_patience, min_delta=early_stop_delta, mode="min"
        )  # Based on minimum validation loss

        # Initialize TensorBoard logger
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("runs/kkt_nn/" + current_time)

        # Initialize metrics dictionary
        self.metrics = {"r2": [], "mape": [], "rmse": [], "mae": []}

        self.losses = {"stationarity": [], "feasibility": [], "complementarity": []}
        # Initialize Sobol engine for parameter sampling
        self.sobol_eng = torch.quasirandom.SobolEngine(
            dimension=self.input_dim, scramble=True, seed=42
        )
        # Load the validation dataset
        self.validation_loader = None
        if validation_filepath is not None:
            self.validation_loader = self.load_validation_dataset(validation_filepath)

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
        validation_loader = DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )
        return validation_loader

    def lagrangian(self, decision_var, dual_eq, dual_ineq, params_dict):
        """
        Computes the Lagrangian for a single sample.

        Args:
            decision_var (torch.Tensor): Denormalized decision variables, shape (num_decision_vars,).
            dual_eq (torch.Tensor or None): Dual variables for equality constraints, shape (num_eq_constraints,).
            dual_ineq (torch.Tensor or None): Dual variables for inequality constraints, shape (num_ineq_constraints,).
            params_dict (dict): Dictionary of parameters {parameter_name: value}.

        Returns:
            torch.Tensor: Scalar value of the Lagrangian.
        """
        # Compute the cost
        lagrangian = self.problem.cost_function(decision_var, params_dict)

        # Add dual terms for equality constraints
        if dual_eq is not None and self.num_eq_constraints > 0:
            expr_eq = []
            for constraint in self.problem.constraints:
                if constraint.type == "equality":
                    expr_eq.append(
                        constraint.get_constraints(
                            decision_var.unsqueeze(0), params_dict
                        )
                    )
            if expr_eq:
                expr_eq = torch.cat(expr_eq, dim=-1).squeeze(0)
                lagrangian += torch.dot(dual_eq, expr_eq)

        # Add dual terms for inequality constraints
        if dual_ineq is not None and self.num_ineq_constraints > 0:
            expr_ineq = []
            for constraint in self.problem.constraints:
                if constraint.type == "inequality":
                    expr_ineq.append(
                        constraint.get_constraints(
                            decision_var.unsqueeze(0), params_dict
                        )
                    )
            if expr_ineq:
                expr_ineq = torch.cat(expr_ineq, dim=-1).squeeze(0)
                lagrangian += torch.dot(dual_ineq, expr_ineq)

        return lagrangian

    def kkt_loss(self, decision_vars, dual_eq_vars, dual_ineq_vars, params_dict):
        """
        Computes the loss based on the KKT conditions using vmap and grad from Functorch.

        Args:
            decision_vars (torch.Tensor): Denormalized decision variables, shape (batch_size, num_decision_vars).
            dual_eq_vars (torch.Tensor or None): Dual variables for equality constraints, shape (batch_size, num_eq_constraints).
            dual_ineq_vars (torch.Tensor or None): Dual variables for inequality constraints, shape (batch_size, num_ineq_constraints).
            params_dict (dict): Dictionary of parameters {parameter_name: value}.

        Returns:
            tuple: (stationarity_loss, feasibility_loss, complementarity_loss)
        """
        batch_size = decision_vars.shape[0]

        # If dual variables are None, replace them with zeros
        if dual_eq_vars is None and self.num_eq_constraints > 0:
            dual_eq_vars = torch.zeros(
                (batch_size, self.num_eq_constraints), device=self.device
            )
        elif self.num_eq_constraints == 0:
            dual_eq_vars = None  # No equality constraints

        if dual_ineq_vars is None and self.num_ineq_constraints > 0:
            dual_ineq_vars = torch.zeros(
                (batch_size, self.num_ineq_constraints), device=self.device
            )
        elif self.num_ineq_constraints == 0:
            dual_ineq_vars = None  # No inequality constraints

        # Determine the in_dims for vmap based on whether dual variables are None
        if dual_eq_vars is not None and dual_ineq_vars is not None:
            in_dims = (0, 0, 0, 0)
        elif dual_eq_vars is None and dual_ineq_vars is not None:
            in_dims = (0, None, 0, 0)
        elif dual_eq_vars is not None and dual_ineq_vars is None:
            in_dims = (0, 0, None, 0)
        else:
            in_dims = (0, None, None, 0)

        # Vectorize the gradient function of the Lagrangian
        grad_L = vmap(grad(self.lagrangian, argnums=0), in_dims=in_dims)(
            decision_vars, dual_eq_vars, dual_ineq_vars, params_dict
        )

        # Compute stationarity loss: sum of squares of gradients
        loss_stationarity = torch.square(grad_L).mean(-1)

        # Compute feasibility loss: sum of squares of constraint violations
        feasibility_loss = 0.0
        for constraint in self.problem.constraints:
            expr = constraint.get_constraints(
                decision_vars, params_dict
            )  # Shape: (batch_size, n_constraints)
            if constraint.type == "equality":
                
                feasibility_loss += torch.square(expr).mean(-1)
            elif constraint.type == "inequality":
                if expr.ndim < 2: print(expr.ndim)
                feasibility_loss += torch.square(torch.relu(expr)).mean(-1)

        # Compute complementarity loss: sum of squares of dual * constraint expressions
        complementarity_loss = 0.0
        if dual_ineq_vars is not None and self.num_ineq_constraints > 0:
            expr_ineq = []
            for constraint in self.problem.constraints:
                if constraint.type == "inequality":
                    expr_ineq.append(
                        constraint.get_constraints(decision_vars, params_dict)
                    )
            if expr_ineq:
                expr_ineq = torch.cat(expr_ineq, dim=-1)

                 # Define a small tolerance epsilon
                epsilon = 1e-6
                # Create masks
                active_constraints = torch.abs(expr_ineq) >= -epsilon
                inactive_constraints = torch.abs(expr_ineq) < -epsilon
                complementarity = (
                    dual_ineq_vars * expr_ineq
                )  # Shape: (batch_size, num_ineq_constraints)
                complementarity_loss += torch.square(complementarity).mean(-1)

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
        norm_params = self.sample_parameters(
            batch_size
        )  # Shape: (batch_size, num_parameters)

        # Denormalize parameters to their original scale
        params_dict = self.problem.denormalize_parameters(norm_params)

        # Forward pass through the model
        norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(norm_params)

        # Denormalize decision variables
        decision_vars = self.problem.denormalize_decision(
            norm_decision_vars, params_dict
        )

        # Compute KKT-based loss
        stationarity_loss, feasibility_loss, complementarity_loss = self.kkt_loss(
            decision_vars, dual_eq_vars, dual_ineq_vars, params_dict
        )
        loss = (stationarity_loss + feasibility_loss + complementarity_loss).mean()

        self.optimizer.zero_grad()
        #backward([stationarity_loss, feasibility_loss, complementarity_loss], self.model.parameters(), UPGrad())
        loss.backward()
        # Optionally clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return (
            loss.item(),
            stationarity_loss.mean().item(),
            feasibility_loss.mean().item(),
            complementarity_loss.mean().item(),
        )

    def validation_step(self, validation_loader):
        """
        Performs a validation step over the entire validation dataset.

        Args:
            validation_loader (DataLoader): DataLoader for the validation dataset.

        Returns:
            tuple: (average R2 score, average MAPE, average RMSE)
        """
        self.model.eval()
        r2_scores = []
        mapes = []
        rmses = []
        maes = []
        with torch.no_grad():
            for params, solutions in validation_loader:
                params = params.to(self.device)
                solutions = solutions.to(self.device)
                # Create params_dict
                params_dict = {}
                for i, var in enumerate(self.problem.parameters):
                    params_dict[var.name] = params[:, i]
                # Normalize parameters
                params_norm = self.problem.normalize_parameters(params)

                # Forward pass through the model
                norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(
                    params_norm
                )

                # Denormalize decision variables
                decision_vars = self.problem.denormalize_decision(
                    norm_decision_vars, params_dict
                )

                # Compute metrics
                r2 = R2Score(self.num_decision_vars, multioutput="variance_weighted").to(self.device)(decision_vars, solutions)
                mape = MeanAbsolutePercentageError().to(self.device)(
                    decision_vars, solutions
                )
                rmse = MeanSquaredError(squared=False).to(self.device)(
                    decision_vars, solutions
                )
                mae = MeanAbsoluteError().to(self.device)(decision_vars, solutions)
                r2_scores.append(r2.item())
                mapes.append(mape.item())
                rmses.append(rmse.item())
                maes.append(mae.item())

        # Calculate average metrics
        avg_r2 = np.mean(r2_scores)
        avg_mape = np.mean(mapes)
        avg_rmse = np.mean(rmses)
        avg_mae = np.mean(maes)
        return avg_r2, avg_mape, avg_rmse, avg_mae

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

    def train_model(self, num_steps, batch_size, checkpoint_interval=None):
        """
        Initiates the training process.

        Args:
            num_steps (int): Total number of training steps.
            batch_size (int): Size of each training batch.
            checkpoint_interval (int, optional): Interval of steps to save checkpoints. Defaults to None.
        """
        for step in tqdm(range(1, num_steps + 1)):
            # Perform a training step
            train_loss, stationarity_loss, feasibility_loss, complementarity_loss = (
                self.training_step(batch_size)
            )
            self.tb_logger.add_scalar("Train/Loss", train_loss, step)
            self.losses["stationarity"].append(stationarity_loss)
            self.losses["feasibility"].append(feasibility_loss)
            self.losses["complementarity"].append(complementarity_loss)
            self.scheduler.step(train_loss)

            # Check for early stopping
            if self.es.early_stop_triggered(train_loss):
                print("Early stopping triggered")
                break

            # Perform validation periodically
            if self.validation_loader and (step % 100 == 0 or step == num_steps):
                r2, mape, rmse, mae = self.validation_step(self.validation_loader)
                self.metrics["r2"].append(r2)
                self.metrics["mape"].append(mape)
                self.metrics["rmse"].append(rmse)
                self.metrics["mae"].append(mae)
                self.tb_logger.add_scalar("Val/R2", r2, step)
                self.tb_logger.add_scalar("Val/MAPE", mape, step)
                self.tb_logger.add_scalar("Val/RMSE", rmse, step)
                self.tb_logger.add_scalar("Val/MAE", mae, step)
                print(
                    f"Step={step}  LR={self.scheduler.optimizer.param_groups[0]['lr']} Train Loss={train_loss:.6f}, Stationarity Loss={stationarity_loss:.6f}, Feasibility Loss={feasibility_loss:.6f}, Complementarity Loss={complementarity_loss:.6f}, Val R2={r2:.4f}, MAPE={mape:.4f}, RMSE={rmse:.6f}, , MAE={mae:.6f}"
                )

            # Save checkpoints at specified intervals
            if (checkpoint_interval) and (step % checkpoint_interval == 0):
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
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": self.metrics,
            "losses": self.losses,
            "es_state": {"counter": self.es.counter, "best": self.es.best},
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

    def save_losses(self, filepath):
        """
        Saves the training losses to a CSV file.

        Args:
            filepath (str): Path to save the losses CSV.
        """
        pd.DataFrame(self.losses).to_csv(filepath, index=False)

    def predict(self, input_params):
        """
        Computes the optimal solution given unnormalized input parameters.

        Args:
            input_params (list or np.ndarray or torch.Tensor): Unnormalized input parameters.

        Returns:
            np.ndarray: Optimal decision variables.
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
            norm_params = self.problem.normalize_parameters(input_params)

            # Forward pass through the model
            norm_decision_vars, dual_eq_vars, dual_ineq_vars = self.model(norm_params)

            # Denormalize parameters
            denorm_params = self.problem.denormalize_parameters(norm_params)
            # Denormalize decision variables
            decision_vars = self.problem.denormalize_decision(
                norm_decision_vars, denorm_params
            )

        return decision_vars.squeeze(0).cpu().numpy()

    def load_checkpoint(self, filepath):
        """
        Loads a model checkpoint.

        Args:
            filepath (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.metrics = checkpoint["metrics"]
        self.losses = checkpoint["losses"]
        self.es.counter = checkpoint["es_state"]["counter"]
        self.es.best = checkpoint["es_state"]["best"]
        self.model.to(self.device)
