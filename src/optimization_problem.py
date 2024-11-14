# optimization_problem.py

import torch
from collections import OrderedDict
from src.kkt_optimizer import KKTOptimizer  # Ensure correct import
from src.decorators import count_constraints  # If using decorators for constraints
from src.utils import transform_variables

class OptimizationProblem:
    """
    OptimizationProblem defines an optimization problem with variables, a cost function,
    and constraints. It interfaces with the KKTOptimizer to solve the problem.
    """

    def __init__(self):
        self.variables = OrderedDict()
        self.cost_function = None
        self.inequality_constraints = []
        self.equality_constraints = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = None  # Will hold the KKTOptimizer instance
        self.total_inequality_constraints = 0  # Counter for inequality constraints
        self.total_equality_constraints = 0  # Counter for equality constraints

    def add_variable(self, name, dim, bounds):
        """
        Adds a variable to the optimization problem.

        Args:
            name (str): The name of the variable.
            dim (int): The dimension of the variable.
            bounds (tuple): A tuple containing the lower and upper bounds of the variable.
                            Bounds can be constants or callables that depend on other variables.
        """
        self.variables[name] = {
            'dim': dim,
            'bounds': bounds,
        }

    def set_cost_function(self, cost_function):
        """
        Sets the cost function of the optimization problem.

        Args:
            cost_function (callable): A function that computes the cost given a dictionary of variables.
        """
        self.cost_function = cost_function

    def add_inequality_constraint(self, constraint_func):
        """
        Adds an inequality constraint to the optimization problem.

        The constraint function must return a tensor of shape [batch_size, num_constraints].

        Args:
            constraint_func (callable): A function that returns the inequality constraints.
        """
        num_constraints = self._determine_num_constraints(constraint_func)
        self.inequality_constraints.append(constraint_func)
        self.total_inequality_constraints += num_constraints

    def add_equality_constraint(self, constraint_func):
        """
        Adds an equality constraint to the optimization problem.

        The constraint function must return a tensor of shape [batch_size, num_constraints].

        Args:
            constraint_func (callable): A function that returns the equality constraints.
        """
        num_constraints = self._determine_num_constraints(constraint_func)
        self.equality_constraints.append(constraint_func)
        self.total_equality_constraints += num_constraints

    def _determine_num_constraints(self, constraint_func):
        """
        Automatically determines the number of constraints returned by a constraint function.

        It uses the 'count_constraints' decorator if present; otherwise, it performs a trial run.

        Args:
            constraint_func (callable): The constraint function.

        Returns:
            int: The number of constraints returned by the function.

        Raises:
            ValueError: If the constraint function does not return a 2D PyTorch tensor.
        """
        if hasattr(constraint_func, 'num_constraints'):
            # Use the number specified by the decorator
            return constraint_func.num_constraints
        else:
            # Fallback: perform a trial run with dummy variables
            # Create a dictionary of dummy variables with batch_size=1
            dummy_variable_values = {}
            for name, var_info in self.variables.items():
                dim = var_info['dim']
                # Set dummy values to 0 for all variables
                dummy_variable_values[name] = torch.zeros((1, dim), device=self.device)

            try:
                # Call the constraint function with dummy variables
                dummy_constraints = constraint_func(dummy_variable_values)
                if not isinstance(dummy_constraints, torch.Tensor):
                    raise TypeError("Constraint function must return a PyTorch tensor.")
                if dummy_constraints.ndim != 2:
                    raise ValueError("Constraint tensor must have 2 dimensions [batch_size, num_constraints].")
                num_constraints = dummy_constraints.shape[1]
                return num_constraints
            except Exception as e:
                raise ValueError(f"Error executing constraint function: {e}")

    def solve(self, epochs=100, batch_size=64, learning_rate=1e-3):
        """
        Solves the optimization problem using the KKTOptimizer.

        Args:
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Number of samples per batch. Defaults to 64.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

        Raises:
            ValueError: If essential components like the solution variable or cost function are not defined.
        """
        # Validation checks
        if 'solution' not in self.variables:
            raise ValueError("The 'solution' variable must be defined in the problem.")
        if self.cost_function is None:
            raise ValueError("The cost function must be defined using set_cost_function().")
        if not self.inequality_constraints and not self.equality_constraints:
            raise ValueError("At least one constraint (inequality or equality) must be added to the problem.")

        # Initialize the KKTOptimizer
        self.optimizer = KKTOptimizer(
            problem=self,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        self.optimizer.optimize()

    def get_solution(self, **input_params):
        """
        Obtains the optimized solution given a set of input parameters.

        Args:
            **input_params: Input parameters for the independent variables.

        Returns:
            numpy.ndarray: The optimized solution.

        Raises:
            ValueError: If the problem has not been solved yet.
        """
        # Ensure that the problem has been solved
        if self.optimizer is None:
            raise ValueError("The problem has not been solved yet. Call the 'solve()' method first.")

        # Prepare input variables
        variable_values = {}
        for name in self.optimizer.input_variables:
            if name in input_params:
                # Use provided values for input variables
                variable_values[name] = torch.tensor(
                    input_params[name], dtype=torch.float32, device=self.device
                )
            else:
                # Generate random values within the bounds for input variables
                var_info = self.variables[name]
                x_min, x_max = var_info['bounds']
                x_min = x_min(self.variables) if callable(x_min) else x_min
                x_max = x_max(self.variables) if callable(x_max) else x_max
                variable_values[name] = torch.rand(
                    (1, var_info['dim']), device=self.device
                ) * (x_max - x_min) + x_min

        # Normalize the input variables
        normalized_vars = transform_variables(
            variable_values, self.variables, mode='normalize'
        )

        # Concatenate all normalized input variables into a single tensor
        inputs = torch.cat(
            [normalized_vars[name] for name in self.optimizer.input_variables], dim=1
        )

        # Forward pass through the network to get the solution and multipliers
        outputs = self.optimizer.kkt_net(inputs)
        solution_norm = outputs[0]
        inequality_multipliers = outputs[1] if self.optimizer.num_inequality_constraints > 0 else None
        equality_multipliers = outputs[2] if self.optimizer.num_equality_constraints > 0 else None

        # Denormalize the solution
        solution_info = self.variables['solution']
        x_min, x_max = solution_info['bounds']
        x_min = x_min(input_params) if callable(x_min) else x_min
        x_max = x_max(input_params) if callable(x_max) else x_max
        solution = (solution_norm + 1) * (x_max - x_min) / 2 + x_min

        return solution.detach().cpu().numpy()