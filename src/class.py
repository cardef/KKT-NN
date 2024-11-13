import torch
from torch import nn, optim
from torch.func import grad, vmap
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Functions for normalizing and denormalizing variables
def normalize_variable(x, x_min, x_max):
    """Normalize variable x to the range [-1, 1] using min and max bounds."""
    return 2 * (x - x_min) / (x_max - x_min) - 1

def denormalize_variable(x_norm, x_min, x_max):
    """Denormalize variable x_norm from the range [-1, 1] back to original scale."""
    return (x_norm + 1) * (x_max - x_min) / 2 + x_min

# Main class for defining the optimization problem
class OptimizationProblem:
    def __init__(self):
        # Dictionary to store variables with their properties
        self.variables = {}
        # List to maintain the order of variables (important for input preparation)
        self.variable_order = []
        # Placeholder for the cost function
        self.cost_function = None
        # Lists to store inequality and equality constraints
        self.inequality_constraints = []
        self.equality_constraints = []
        # Dictionary to store variable dependencies
        self.variable_dependencies = {}
        # Device configuration (CPU or GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_variable(self, name, dim, bounds, dependencies=None):
        """
        Add a variable to the optimization problem.

        Parameters:
        - name (str): Name of the variable.
        - dim (int): Dimension of the variable.
        - bounds (tuple): Tuple containing min and max bounds. Bounds can be functions.
        - dependencies (list): List of variable names this variable depends on.
        """
        self.variables[name] = {
            'dim': dim,
            'bounds': bounds,
            'dependencies': dependencies or []
        }
        self.variable_order.append(name)
        if dependencies:
            self.variable_dependencies[name] = dependencies

    def set_cost_function(self, cost_function):
        """
        Set the cost function of the optimization problem.

        Parameters:
        - cost_function (function): A function that computes the cost given variable values.
        """
        self.cost_function = cost_function

    def add_inequality_constraint(self, constraint_function):
        """
        Add an inequality constraint to the optimization problem.

        Parameters:
        - constraint_function (function): Function that computes the inequality constraints.
        """
        self.inequality_constraints.append(constraint_function)

    def add_equality_constraint(self, constraint_function):
        """
        Add an equality constraint to the optimization problem.

        Parameters:
        - constraint_function (function): Function that computes the equality constraints.
        """
        self.equality_constraints.append(constraint_function)

    def solve(self, epochs=100, batch_size=64, learning_rate=1e-3):
        """
        Solve the optimization problem by training the neural network.

        Parameters:
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = KKTOptimizer(
            problem=self,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        optimizer.optimize()
        self.optimizer = optimizer  # Save the optimizer for future use

    def get_solution(self, **input_params):
        """
        Get the optimized solution for given input parameters.

        Parameters:
        - input_params (dict): Dictionary of input variable values.

        Returns:
        - solution (numpy array): Optimized solution.
        """
        return self.optimizer.get_solution(**input_params)

# Internal class for training the neural network using KKT conditions
class KKTOptimizer:
    def __init__(self, problem, batch_size=64, learning_rate=1e-3, epochs=100):
        self.problem = problem  # Reference to the optimization problem
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = problem.device
        self.build_model()  # Build the neural network model

        # Initialize the optimizer (Adam optimizer in this case)
        self.optimizer = optim.Adam(self.kkt_net.parameters(), lr=self.learning_rate)
        # Sobol sequence generator for sampling
        self.sobol_engine = torch.quasirandom.SobolEngine(
            self.total_input_dim, scramble=True, seed=42
        )
        # Setup for TensorBoard logging
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tb_logger = SummaryWriter("logs/" + current_time)

    def build_model(self):
        """
        Build the neural network model, including shared layers, solution layers,
        and multiplier layers for inequality and equality constraints.
        """
        # Calculate the total input dimension by summing the dimensions of all variables
        self.input_dims = {name: var['dim'] for name, var in self.problem.variables.items()}
        self.total_input_dim = sum(self.input_dims.values())

        # Define the shared layers of the neural network
        self.shared_layers = nn.Sequential(
            nn.Linear(self.total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        # Define the solution layers (output layers for the solution variable)
        solution_dim = self.problem.variables['solution']['dim']
        self.solution_layers = nn.Sequential(
            nn.Linear(64, solution_dim),
            nn.Tanh()  # Output activation to ensure values are in [-1, 1]
        )
        # Determine the total number of inequality constraints
        self.num_inequality_constraints = 0
        if self.problem.inequality_constraints:
            # To find the number of constraints, perform a dummy forward pass
            sample_input = torch.zeros((1, self.total_input_dim))
            variable_values = self.prepare_forward(sample_input)
            with torch.no_grad():
                constraints = self.compute_inequality_constraints(variable_values)
                self.num_inequality_constraints = constraints.shape[1]
        else:
            self.num_inequality_constraints = 0

        # Define the multiplier layers for inequality constraints if any
        if self.num_inequality_constraints > 0:
            self.inequality_multiplier_layers = nn.Sequential(
                nn.Linear(64, self.num_inequality_constraints),
                nn.Softplus()  # Activation to ensure multipliers are non-negative
            )
        else:
            self.inequality_multiplier_layers = None

        # Equality multiplier layers (not used in this example)
        self.equality_multiplier_layers = None

        # Define the KKT neural network with the layers
        self.kkt_net = KKTNet(
            self.shared_layers,
            self.solution_layers,
            self.inequality_multiplier_layers,
            self.equality_multiplier_layers
        ).to(self.device)

    def prepare_batch(self):
        """
        Generate a batch of data by sampling from the Sobol sequence.

        Returns:
        - variable_values (dict): Dictionary of variables with their sampled values.
        """
        # Generate a batch of samples in [0, 1]^d space
        sampled_batch = self.sobol_engine.draw(self.batch_size).to(self.device)
        # Split the sampled batch into individual variables
        variable_values = {}
        idx = 0
        for name in self.problem.variable_order:
            dim = self.problem.variables[name]['dim']
            variable_values[name] = sampled_batch[:, idx:idx+dim]
            idx += dim
        return variable_values

    def normalize_variables(self, variable_values):
        """
        Normalize the variables to the range [-1, 1] based on their bounds.

        Parameters:
        - variable_values (dict): Dictionary of variables with their sampled values.

        Returns:
        - normalized_vars (dict): Dictionary of normalized variables.
        """
        normalized_vars = {}
        params = {}
        for name in self.problem.variable_order:
            var_info = self.problem.variables[name]
            x_min, x_max = var_info['bounds']
            # Handle callable bounds (dependencies)
            x_min = x_min if not callable(x_min) else x_min(params)
            x_max = x_max if not callable(x_max) else x_max(params)
            var_value = variable_values[name]
            # Normalize the variable
            normalized_var = normalize_variable(var_value, x_min, x_max)
            normalized_vars[name] = normalized_var
            params[name] = var_value  # Update params for dependent variables
        return normalized_vars

    def denormalize_variables(self, normalized_vars):
        """
        Denormalize the variables back to their original scale based on their bounds.

        Parameters:
        - normalized_vars (dict): Dictionary of normalized variables.

        Returns:
        - denorm_vars (dict): Dictionary of denormalized variables.
        """
        denorm_vars = {}
        params = {}
        for name in self.problem.variable_order:
            var_info = self.problem.variables[name]
            x_min, x_max = var_info['bounds']
            # Handle callable bounds (dependencies)
            if callable(x_min):
                x_min = x_min(params)
            if callable(x_max):
                x_max = x_max(params)
            var_value = normalized_vars[name]
            # Denormalize the variable
            denorm_var = denormalize_variable(var_value, x_min, x_max)
            denorm_vars[name] = denorm_var
            params[name] = denorm_var  # Update params for dependent variables
        return denorm_vars

    def compute_inequality_constraints(self, variable_values):
        """
        Compute the inequality constraints given the variable values.

        Parameters:
        - variable_values (dict): Dictionary of variables.

        Returns:
        - constraints (tensor): Tensor of inequality constraint values.
        """
        constraints_list = []
        for constraint_func in self.problem.inequality_constraints:
            constraints = constraint_func(variable_values)
            constraints_list.append(constraints)
        if constraints_list:
            # Concatenate constraints from all functions
            return torch.cat(constraints_list, dim=1)
        else:
            return None

    def compute_equality_constraints(self, variable_values):
        """
        Compute the equality constraints given the variable values.

        Parameters:
        - variable_values (dict): Dictionary of variables.

        Returns:
        - constraints (tensor): Tensor of equality constraint values.
        """
        constraints_list = []
        for constraint_func in self.problem.equality_constraints:
            constraints = constraint_func(variable_values)
            constraints_list.append(constraints)
        if constraints_list:
            # Concatenate constraints from all functions
            return torch.cat(constraints_list, dim=1)
        else:
            return None

    def optimize(self):
        """
        Train the neural network to approximate the KKT conditions and solve the optimization problem.
        """
        num_epochs = self.epochs
        for epoch in range(num_epochs):
            self.kkt_net.train()
            # Prepare a batch of variable values
            variable_values = self.prepare_batch()
            # Normalize variables
            normalized_vars = self.normalize_variables(variable_values)
            # Create the input tensor for the neural network
            inputs = torch.cat([normalized_vars[name] for name in self.problem.variable_order], dim=1)
            # Forward pass through the neural network
            outputs = self.kkt_net(inputs)
            solution_norm = outputs[0]
            inequality_multipliers = outputs[1]

            # Denormalize variables
            denorm_vars = self.denormalize_variables(normalized_vars)
            # Denormalize the solution variable
            denorm_vars['solution'] = denormalize_variable(
                solution_norm,
                *self.problem.variables['solution']['bounds']
            )

            # Prepare variables for cost and constraint functions
            variables_for_functions = denorm_vars.copy()
            variables_for_functions['inequality_multipliers'] = inequality_multipliers

            # Compute the loss based on KKT conditions
            loss = self.kkt_loss(solution_norm, inequality_multipliers, variables_for_functions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def kkt_loss(self, solution_norm, inequality_multipliers, variables):
        """
        Compute the total loss based on the KKT conditions.

        Parameters:
        - solution_norm (tensor): Normalized solution from the neural network.
        - inequality_multipliers (tensor): Multipliers for inequality constraints.
        - variables (dict): Dictionary of variables.

        Returns:
        - loss (tensor): Scalar tensor representing the total loss.
        """
        # Extract the solution variable
        solution = variables['solution']

        # Define the lagrangian function for gradient computation
        def lagrangian_func(solution, inequality_multipliers, variables):
            variables['solution'] = solution
            cost = self.problem.cost_function(variables)
            lagrangian = cost
            # Add inequality constraints to the lagrangian
            if inequality_multipliers is not None and self.problem.inequality_constraints:
                inequality_constraints = self.compute_inequality_constraints(variables)
                lagrangian += torch.sum(inequality_multipliers * inequality_constraints, dim=1)
            return lagrangian

        # Compute the gradient of the lagrangian with respect to the solution
        grad_lagrangian = vmap(grad(lagrangian_func, argnums=0), in_dims=(0, 0, None))(
            solution, inequality_multipliers, variables
        )

        # Stationarity condition loss (norm squared of gradient)
        loss_stationarity = torch.square(grad_lagrangian).sum(dim=1).mean()

        # Primal feasibility conditions
        inequality_constraints = self.compute_inequality_constraints(variables)
        if inequality_constraints is not None:
            # Feasibility loss for inequality constraints
            loss_feasibility = torch.square(torch.relu(inequality_constraints)).sum(dim=1).mean()
            # Complementarity condition loss
            complementarity = inequality_multipliers * inequality_constraints
            loss_complementarity = torch.square(complementarity).sum(dim=1).mean()
        else:
            loss_feasibility = torch.tensor(0.0, device=self.device)
            loss_complementarity = torch.tensor(0.0, device=self.device)

        # Total loss is the sum of stationarity, feasibility, and complementarity losses
        loss = loss_stationarity + loss_feasibility + loss_complementarity
        return loss

    def get_solution(self, **input_params):
        """
        Obtain the optimized solution given input parameters.

        Parameters:
        - input_params (dict): Dictionary of input variable values.

        Returns:
        - solution (numpy array): Optimized solution as a numpy array.
        """
        # Prepare input variables
        variable_values = {}
        for name in self.problem.variable_order:
            if name in input_params:
                # Use provided input values
                variable_values[name] = torch.tensor(input_params[name], dtype=torch.float32, device=self.device)
            else:
                # Generate random values within bounds if not provided
                var_info = self.problem.variables[name]
                x_min, x_max = var_info['bounds']
                x_min = x_min if not callable(x_min) else x_min(variable_values)
                x_max = x_max if not callable(x_max) else x_max(variable_values)
                variable_values[name] = torch.rand(var_info['dim'], device=self.device) * (x_max - x_min) + x_min

        # Normalize variables
        normalized_vars = self.normalize_variables(variable_values)
        # Create the input tensor for the neural network
        inputs = torch.cat([normalized_vars[name].unsqueeze(0) for name in self.problem.variable_order], dim=1)
        # Forward pass through the neural network
        outputs = self.kkt_net(inputs)
        solution_norm = outputs[0]
        # Denormalize the solution variable
        solution = denormalize_variable(
            solution_norm,
            *self.problem.variables['solution']['bounds']
        )
        return solution.detach().cpu().numpy()

# Neural network class for KKT conditions
class KKTNet(nn.Module):
    def __init__(self, shared_net, solution_net, inequality_multiplier_net=None, equality_multiplier_net=None):
        super().__init__()
        # Shared layers
        self.shared = shared_net
        # Solution layers
        self.solution = solution_net
        # Multiplier layers for inequality constraints
        self.inequality_multiplier_net = inequality_multiplier_net
        # Multiplier layers for equality constraints (not used in this example)
        self.equality_multiplier_net = equality_multiplier_net

    def forward(self, inputs):
        """
        Forward pass of the neural network.

        Parameters:
        - inputs (tensor): Input tensor containing all variables.

        Returns:
        - outputs (list): List containing solution and multipliers.
        """
        embedding = self.shared(inputs)
        solution = self.solution(embedding)
        outputs = [solution]

        if self.inequality_multiplier_net is not None:
            inequality_multipliers = self.inequality_multiplier_net(embedding)
            outputs.append(inequality_multipliers)
        else:
            outputs.append(None)

        if self.equality_multiplier_net is not None:
            equality_multipliers = self.equality_multiplier_net(embedding)
            outputs.append(equality_multipliers)
        else:
            outputs.append(None)

        return outputs
