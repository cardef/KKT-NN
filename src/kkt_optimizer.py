# src/kkt_optimizer.py

import torch
from torch import nn, optim
from torch.func import grad, vmap

from src.utils import transform_variables
from src.kkt_network import KKTNet


class KKTOptimizer:
    """
    KKTOptimizer is responsible for optimizing an optimization problem using
    the Karush-Kuhn-Tucker (KKT) conditions. It utilizes a neural network to
    approximate the solution and the associated Lagrange multipliers.
    """

    def __init__(self, problem, batch_size=64, learning_rate=1e-3, epochs=100):
        """
        Initializes the KKTOptimizer.

        Args:
            problem (OptimizationProblem): The optimization problem to solve.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            epochs (int, optional): Number of training epochs. Defaults to 100.
        """
        self.problem = problem
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = problem.device

        # Identify independent variables (exclude 'solution')
        self.input_variables = [
            name for name in self.problem.variables.keys() if name != 'solution'
        ]
        self.input_dims = {
            name: var_info['dim'] for name, var_info in self.problem.variables.items() if name != 'solution'
        }
        self.total_input_dim = sum(self.input_dims.values())

        # Sorted list of variable names for consistent ordering
        self.sorted_variable_names = sorted(self.input_variables)

        # Build the model after defining input dimensions
        self.build_model()

        # Initialize the Adam optimizer with the neural network's parameters
        self.adam_optimizer = optim.Adam(self.kkt_net.parameters(), lr=self.learning_rate)

        # Initialize a Sobol Engine for quasi-random sampling of independent variables
        self.sobol_engine = torch.quasirandom.SobolEngine(
            self.total_input_dim, scramble=True, seed=42
        )

    def build_model(self):
        """
        Builds the neural network model based on the problem's variables and constraints.
        """
        # Define shared layers for feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Define layers to predict the normalized solution
        solution_dim = self.problem.variables['solution']['dim']
        self.solution_layers = nn.Sequential(
            nn.Linear(64, solution_dim),
            nn.Tanh()  # Ensures the output is within a specific range
        )

        # Retrieve the number of inequality and equality constraints
        self.num_inequality_constraints = self.problem.total_inequality_constraints
        self.num_equality_constraints = self.problem.total_equality_constraints

        # Define layers for inequality multipliers if any
        if self.num_inequality_constraints > 0:
            self.inequality_multiplier_layers = nn.Sequential(
                nn.Linear(64, self.num_inequality_constraints),
                nn.Softplus()  # Ensures multipliers are non-negative
            )
        else:
            self.inequality_multiplier_layers = None

        # Define layers for equality multipliers if any
        if self.num_equality_constraints > 0:
            self.equality_multiplier_layers = nn.Linear(64, self.num_equality_constraints)
            # Equality multipliers can take any real value; no activation function needed
        else:
            self.equality_multiplier_layers = None

        # Instantiate KKTNet with the defined layers
        self.kkt_net = KKTNet(
            self.shared_layers,
            self.solution_layers,
            self.inequality_multiplier_layers,
            self.equality_multiplier_layers
        ).to(self.device)

    def split_tensor_into_variables(self, tensor):
        """
        Splits a concatenated tensor into individual variables based on the problem's definition.

        Args:
            tensor (torch.Tensor): The concatenated tensor of all input variables.

        Returns:
            dict: A dictionary mapping variable names to their respective tensors.
        """
        variables = {}
        idx = 0
        for name in self.sorted_variable_names:
            dim = self.input_dims[name]
            variables[name] = tensor[:, idx:idx + dim]
            idx += dim
        return variables

    def prepare_batch(self):
        """
        Samples a batch of input variables using the Sobol Engine.

        Returns:
            dict: A dictionary of sampled variable tensors.
        """
        sampled_batch = self.sobol_engine.draw(self.batch_size).to(self.device)
        variable_values = self.split_tensor_into_variables(sampled_batch)
        return variable_values

    def optimize(self):
        """
        Trains the neural network to minimize the KKT loss over the specified number of epochs.
        """
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            self.kkt_net.train()  # Set the network to training mode

            # Prepare a batch of variable values
            variable_values = self.prepare_batch()

            # Normalize the variables
            normalized_vars = transform_variables(
                variable_values, self.problem.variables, mode='normalize'
            )

            # Concatenate all normalized input variables into a single tensor
            inputs = torch.cat(
                [normalized_vars[name] for name in self.sorted_variable_names],
                dim=1
            )

            # Forward pass through the network
            outputs = self.kkt_net(inputs)
            solution_norm = outputs[0]
            inequality_multipliers = outputs[1] if self.num_inequality_constraints > 0 else None
            equality_multipliers = outputs[2] if self.num_equality_constraints > 0 else None

            # Prepare variables for loss computation
            variables = normalized_vars.copy()
            variables['solution_norm'] = solution_norm
            if inequality_multipliers is not None:
                variables['inequality_multipliers'] = inequality_multipliers
            if equality_multipliers is not None:
                variables['equality_multipliers'] = equality_multipliers

            # Compute the KKT-based loss
            loss = self.kkt_loss(variables)

            # Backpropagation and optimizer step
            self.adam_optimizer.zero_grad()
            loss.backward()
            self.adam_optimizer.step()

            # Log progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    def kkt_loss(self, variables):
        """
        Computes the loss based on the KKT conditions.

        Args:
            variables (dict): A dictionary containing normalized variables and multipliers.

        Returns:
            torch.Tensor: The computed loss.
        """
        # Extract solution and multipliers from variables
        solution_norm = variables.pop('solution_norm')  # [batch_size, solution_dim]
        inequality_multipliers = variables.pop('inequality_multipliers', None)  # [batch_size, num_ineq]
        equality_multipliers = variables.pop('equality_multipliers', None)  # [batch_size, num_eq]

        # Prepare variables to denormalize (only input variables)
        variables_to_transform = {k: v for k, v in variables.items() if k in self.input_variables}
        variables_info_to_transform = {
            k: v for k, v in self.problem.variables.items() if k in self.input_variables
        }

        # Denormalize input variables
        variables_denorm = transform_variables(
            variables_to_transform,
            variables_info_to_transform,
            mode='denormalize'
        )

        # Denormalize the solution
        solution_info = self.problem.variables['solution']
        x_min, x_max = solution_info['bounds']

        # Handle bounds if they are callable
        if callable(x_min):
            x_min = x_min(variables_denorm)
        if callable(x_max):
            x_max = x_max(variables_denorm)

        # Convert bounds to tensors if they are lists or scalars
        if isinstance(x_min, list):
            x_min = torch.tensor(x_min, dtype=solution_norm.dtype, device=solution_norm.device)
        elif isinstance(x_min, (int, float)):
            x_min = torch.tensor([x_min], dtype=solution_norm.dtype, device=solution_norm.device)
        elif isinstance(x_min, torch.Tensor):
            if x_min.ndim != 1:
                raise ValueError(f"x_min tensor should be 1-dimensional, got {x_min.ndim} dimensions")
            # x_min remains a tensor
        else:
            raise TypeError(f"Unsupported type for x_min in 'solution' variable: {type(x_min)}")

        if isinstance(x_max, list):
            x_max = torch.tensor(x_max, dtype=solution_norm.dtype, device=solution_norm.device)
        elif isinstance(x_max, (int, float)):
            x_max = torch.tensor([x_max], dtype=solution_norm.dtype, device=solution_norm.device)
        elif isinstance(x_max, torch.Tensor):
            if x_max.ndim != 1:
                raise ValueError(f"x_max tensor should be 1-dimensional, got {x_max.ndim} dimensions")
            # x_max remains a tensor
        else:
            raise TypeError(f"Unsupported type for x_max in 'solution' variable: {type(x_max)}")

        # Prevent division by zero in tau calculations
        denominator = (x_max - x_min)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-6, denominator)

        # Denormalize the solution
        solution = (solution_norm + 1) * (x_max - x_min) / 2 + x_min  # [batch_size, dim]
        variables_denorm['solution'] = solution

        # Convert variables_denorm to a tuple of tensors sorted by variable names
        # Exclude 'solution' since it's already passed separately
        variables_denorm_tuple = tuple(variables_denorm[name] for name in self.sorted_variable_names)

        # Define the Lagrangian function based on available multipliers
        if inequality_multipliers is not None and equality_multipliers is not None:
            # Both multipliers are present
            def lagrangian_func(solution, ineq_mult, eq_mult, *variables_denorm):
                variables_dict = dict(zip(self.sorted_variable_names, variables_denorm))
                variables_dict['solution'] = solution  # Add 'solution' separately
                cost = self.problem.cost_function(variables_dict)  # [batch_size]
                lagrangian = cost  # [batch_size]
                if self.num_inequality_constraints > 0:
                    inequality_constraints = self.compute_inequality_constraints(variables_dict)  # [num_ineq]
                    lagrangian += torch.sum(ineq_mult * inequality_constraints, dim=0)  # [batch_size]
                if self.num_equality_constraints > 0:
                    equality_constraints = self.compute_equality_constraints(variables_dict)  # [num_eq]
                    lagrangian += torch.sum(eq_mult * equality_constraints, dim=0)  # [batch_size]
                return lagrangian  # [batch_size]

            # All arguments have a batch dimension
            in_dims = (0, 0, 0) + (0,) * len(variables_denorm_tuple)
            args = (solution, inequality_multipliers, equality_multipliers) + variables_denorm_tuple

        elif inequality_multipliers is not None:
            # Only inequality multipliers are present
            def lagrangian_func(solution, ineq_mult, *variables_denorm):
                variables_dict = dict(zip(self.sorted_variable_names, variables_denorm))
                variables_dict['solution'] = solution  # Add 'solution' separately
                cost = self.problem.cost_function(variables_dict)  # [batch_size]
                lagrangian = cost  # [batch_size]
                if self.num_inequality_constraints > 0:
                    inequality_constraints = self.compute_inequality_constraints(variables_dict)  # [num_ineq]
                    lagrangian += torch.sum(ineq_mult * inequality_constraints, dim=0)  # [batch_size]
                return lagrangian  # [batch_size]

            # All arguments have a batch dimension
            in_dims = (0, 0) + (0,) * len(variables_denorm_tuple)
            args = (solution, inequality_multipliers) + variables_denorm_tuple

        elif equality_multipliers is not None:
            # Only equality multipliers are present
            def lagrangian_func(solution, eq_mult, *variables_denorm):
                variables_dict = dict(zip(self.sorted_variable_names, variables_denorm))
                variables_dict['solution'] = solution  # Add 'solution' separately
                cost = self.problem.cost_function(variables_dict)  # [batch_size]
                lagrangian = cost  # [batch_size]
                if self.num_equality_constraints > 0:
                    equality_constraints = self.compute_equality_constraints(variables_dict)  # [num_eq]
                    lagrangian += torch.sum(eq_mult * equality_constraints, dim=0)  # [batch_size]
                return lagrangian  # [batch_size]

            # All arguments have a batch dimension
            in_dims = (0, 0) + (0,) * len(variables_denorm_tuple)
            args = (solution, equality_multipliers) + variables_denorm_tuple

        else:
            # No multipliers are present
            def lagrangian_func(solution, *variables_denorm):
                variables_dict = dict(zip(self.sorted_variable_names, variables_denorm))
                variables_dict['solution'] = solution  # Add 'solution' separately
                cost = self.problem.cost_function(variables_dict)  # [batch_size]
                return cost  # [batch_size]

            # All arguments have a batch dimension
            in_dims = (0,) + (0,) * len(variables_denorm_tuple)
            args = (solution,) + variables_denorm_tuple

        # Compute the gradient of the Lagrangian with respect to the solution
        grad_lagrangian = vmap(grad(lagrangian_func, argnums=0), in_dims=in_dims)(*args)  # [batch_size, solution_dim]

        # Loss component for the stationarity condition
        loss_stationarity = torch.square(grad_lagrangian).sum(dim=1).mean()

        # Compute inequality constraints and their corresponding losses
        inequality_constraints = self.compute_inequality_constraints(variables_denorm) if self.num_inequality_constraints > 0 else None
        if inequality_constraints is not None:
            # Ensure inequality_constraints has shape [batch_size, num_ineq]
            loss_feasibility = torch.square(torch.relu(inequality_constraints)).sum(dim=1).mean()
            # Loss component for complementarity condition
            complementarity = inequality_multipliers * inequality_constraints
            loss_complementarity = torch.square(complementarity).sum(dim=1).mean()
        else:
            loss_feasibility = torch.tensor(0.0, device=self.device)
            loss_complementarity = torch.tensor(0.0, device=self.device)

        # Compute equality constraints and their corresponding losses
        equality_constraints = self.compute_equality_constraints(variables_denorm) if self.num_equality_constraints > 0 else None
        if equality_constraints is not None:
            # Loss component for equality constraints (quadratic error)
            loss_equality = torch.square(equality_constraints).sum(dim=1).mean()
        else:
            loss_equality = torch.tensor(0.0, device=self.device)

        # Total loss is the sum of all components
        loss = loss_stationarity + loss_feasibility + loss_complementarity + loss_equality
        return loss

    def compute_inequality_constraints(self, variables):
        """
        Computes all inequality constraints by applying each constraint function.

        Args:
            variables (dict): A dictionary of denormalized variables.

        Returns:
            torch.Tensor or None: Concatenated tensor of all inequality constraints or None if no constraints exist.
        """
        constraints_list = []
        for constraint_func in self.problem.inequality_constraints:
            constraints = constraint_func(variables)  # Shape: [7]
            constraints_list.append(constraints)
        if constraints_list:
            return torch.stack(constraints_list, dim=1)  # Shape: [batch_size, num_ineq]
        else:
            return None

    def compute_equality_constraints(self, variables):
        """
        Computes all equality constraints by applying each constraint function.

        Args:
            variables (dict): A dictionary of denormalized variables.

        Returns:
            torch.Tensor or None: Concatenated tensor of all equality constraints or None if no constraints exist.
        """
        constraints_list = []
        for constraint_func in self.problem.equality_constraints:
            constraints = constraint_func(variables)  # Shape: [1]
            constraints_list.append(constraints)
        if constraints_list:
            return torch.stack(constraints_list, dim=1)  # Shape: [batch_size, num_eq]
        else:
            return None
