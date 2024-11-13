# kkt_optimizer.py
import torch
from torch import nn, optim
from torch.func import grad, vmap
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from utils import transform_variables
from kkt_network import KKTNet
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class KKTOptimizer:
    def __init__(self, problem, batch_size=64, learning_rate=1e-3, epochs=100):
        self.problem = problem
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = problem.device
        self.build_model()

        self.optimizer = optim.Adam(self.kkt_net.parameters(), lr=self.learning_rate)
        self.sobol_engine = torch.quasirandom.SobolEngine(
            self.total_input_dim, scramble=True, seed=42
        )
        self.es = EarlyStopper(patience=1000)
        # Setup for TensorBoard logging (optional)
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_logger = SummaryWriter("logs/" + current_time)

    def build_model(self):
        self.input_dims = {name: var['dim'] for name, var in self.problem.variables.items()}
        self.total_input_dim = sum(self.input_dims.values())

        # Define the neural network layers
        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        solution_dim = self.problem.variables['solution']['dim']
        self.solution_layers = nn.Sequential(
            nn.Linear(64, solution_dim),
            nn.Tanh()
        )

        # Determine the number of inequality constraints
        self.num_inequality_constraints = self.get_num_constraints()

        if self.num_inequality_constraints > 0:
            self.inequality_multiplier_layers = nn.Sequential(
                nn.Linear(64, self.num_inequality_constraints),
                nn.Softplus()
            )
        else:
            self.inequality_multiplier_layers = None

        self.equality_multiplier_layers = None  # Not used in this example

        # Define the KKT neural network
        self.kkt_net = KKTNet(
            self.shared_layers,
            self.solution_layers,
            self.inequality_multiplier_layers,
            self.equality_multiplier_layers
        ).to(self.device)

    def get_num_constraints(self):
        # Perform a dummy forward pass to determine the number of constraints
        sample_input = torch.zeros((1, self.total_input_dim), device=self.device)
        variables = self.prepare_variables(sample_input)
        with torch.no_grad():
            constraints = self.compute_inequality_constraints(variables)
            if constraints is not None:
                return constraints.shape[1]
            else:
                return 0

    def prepare_batch(self):
        sampled_batch = self.sobol_engine.draw(self.batch_size).to(self.device)
        variable_values = {}
        idx = 0
        for name in self.problem.variable_order:
            dim = self.problem.variables[name]['dim']
            variable_values[name] = sampled_batch[:, idx:idx+dim]
            idx += dim
        return variable_values

    def prepare_variables(self, inputs):
        normalized_vars = {}
        idx = 0
        for name in self.problem.variable_order:
            dim = self.problem.variables[name]['dim']
            normalized_vars[name] = inputs[:, idx:idx+dim]
            idx += dim
        return normalized_vars

    def compute_inequality_constraints(self, variables):
        constraints_list = []
        for constraint_func in self.problem.inequality_constraints:
            constraints = constraint_func(variables)
            constraints_list.append(constraints)
        if constraints_list:
            return torch.cat(constraints_list, dim=1)
        else:
            return None

    def compute_equality_constraints(self, variables):
        constraints_list = []
        for constraint_func in self.problem.equality_constraints:
            constraints = constraint_func(variables)
            constraints_list.append(constraints)
        if constraints_list:
            return torch.cat(constraints_list, dim=1)
        else:
            return None

    def optimize(self):
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            self.kkt_net.train()
            # Prepare a batch of variable values
            variable_values = self.prepare_batch()
            # Normalize variables
            normalized_vars = transform_variables(
                variable_values, self.problem.variables, mode='normalize'
            )
            # Create the input tensor for the neural network
            inputs = torch.cat(
                [normalized_vars[name] for name in self.problem.variable_order], dim=1
            )
            # Forward pass through the neural network
            outputs = self.kkt_net(inputs)
            solution_norm = outputs[0]
            inequality_multipliers = outputs[1]

            # Prepare variables for loss computation
            variables = normalized_vars.copy()
            variables['solution_norm'] = solution_norm
            variables['inequality_multipliers'] = inequality_multipliers

            # Compute the loss based on KKT conditions
            loss = self.kkt_loss(variables)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

    def kkt_loss(self, variables):
        # Extract 'solution_norm' and 'inequality_multipliers' from variables
        solution_norm = variables.pop('solution_norm')
        inequality_multipliers = variables.pop('inequality_multipliers')

        # Denormalize all variables
        variables_denorm = transform_variables(
            variables, self.problem.variables, mode='denormalize'
        )

        # Denormalize the solution
        solution_info = self.problem.variables['solution']
        x_min, x_max = solution_info['bounds']
        x_min = x_min if not callable(x_min) else x_min(variables_denorm)
        x_max = x_max if not callable(x_max) else x_max(variables_denorm)
        solution = (solution_norm + 1) * (x_max - x_min) / 2 + x_min
        variables_denorm['solution'] = solution

        # Define the lagrangian function for gradient computation
        def lagrangian_func(solution_norm, inequality_multipliers, variables_norm):
            # Denormalize variables inside the function
            variables_denorm = transform_variables(
                variables_norm, self.problem.variables, mode='denormalize'
            )
            # Denormalize the solution
            x_min, x_max = solution_info['bounds']
            x_min = x_min if not callable(x_min) else x_min(variables_denorm)
            x_max = x_max if not callable(x_max) else x_max(variables_denorm)
            solution = (solution_norm + 1) * (x_max - x_min) / 2 + x_min
            variables_denorm['solution'] = solution

            # Compute the lagrangian
            cost = self.problem.cost_function(variables_denorm)
            lagrangian = cost
            if inequality_multipliers is not None and self.problem.inequality_constraints:
                inequality_constraints = self.compute_inequality_constraints(variables_denorm)
                lagrangian += torch.sum(inequality_multipliers * inequality_constraints, dim=1)
            return lagrangian

        # Compute the gradient of the lagrangian with respect to solution_norm
        grad_lagrangian = vmap(grad(lagrangian_func, argnums=0), in_dims=(0, 0, None))(
            solution_norm, inequality_multipliers, variables
        )

        # Stationarity condition loss
        loss_stationarity = torch.square(grad_lagrangian).sum(dim=1).mean()

        # Compute primal feasibility and complementarity conditions
        inequality_constraints = self.compute_inequality_constraints(variables_denorm)
        if inequality_constraints is not None:
            # Feasibility loss for inequality constraints
            loss_feasibility = torch.square(torch.relu(inequality_constraints)).sum(dim=1).mean()
            # Complementarity condition loss
            complementarity = inequality_multipliers * inequality_constraints
            loss_complementarity = torch.square(complementarity).sum(dim=1).mean()
        else:
            loss_feasibility = torch.tensor(0.0, device=self.device)
            loss_complementarity = torch.tensor(0.0, device=self.device)

        # Total loss
        loss = loss_stationarity + loss_feasibility + loss_complementarity
        return loss

    def get_solution(self, **input_params):
        # Prepare input variables
        variable_values = {}
        for name in self.problem.variable_order:
            if name in input_params:
                # Use provided input values
                variable_values[name] = torch.tensor(
                    input_params[name], dtype=torch.float32, device=self.device
                )
            else:
                # Generate random values within bounds if not provided
                var_info = self.problem.variables[name]
                x_min, x_max = var_info['bounds']
                x_min = x_min if not callable(x_min) else x_min(variable_values)
                x_max = x_max if not callable(x_max) else x_max(variable_values)
                variable_values[name] = torch.rand(
                    var_info['dim'], device=self.device
                ) * (x_max - x_min) + x_min

        # Normalize variables
        normalized_vars = transform_variables(
            variable_values, self.problem.variables, mode='normalize'
        )
        # Create the input tensor for the neural network
        inputs = torch.cat(
            [normalized_vars[name].unsqueeze(0) for name in self.problem.variable_order], dim=1
        )
        # Forward pass through the neural network
        outputs = self.kkt_net(inputs)
        solution_norm = outputs[0]
        # Denormalize the solution
        solution_info = self.problem.variables['solution']
        x_min, x_max = solution_info['bounds']
        x_min = x_min if not callable(x_min) else x_min(variable_values)
        x_max = x_max if not callable(x_max) else x_max(variable_values)
        solution = (solution_norm + 1) * (x_max - x_min) / 2 + x_min
        return solution.detach().cpu().numpy()
