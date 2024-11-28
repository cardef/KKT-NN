import torch

class Variable:
    """
    Represents a variable in the optimization problem, with methods to get its minimum and maximum values,
    which can be constants or functions of other parameters.
    """

    def __init__(self, name, min_val, max_val):
        """
        Initializes the Variable.

        Args:
            name (str): Name of the variable.
            min_val (float or callable): Minimum value or function to compute the minimum.
            max_val (float or callable): Maximum value or function to compute the maximum.
        """
        self.name = name
        self.min_val = min_val
        self.max_val = max_val

    def get_min(self, denorm_params):
        """
        Returns the minimum value of the variable, which can depend on other parameters.

        Args:
            denorm_params (dict): Dictionary of denormalized parameters.

        Returns:
            torch.Tensor: Minimum value of the variable.
        """
        if callable(self.min_val):
            return self.min_val(denorm_params)
        else:
            if denorm_params is None or not denorm_params:
                return torch.tensor(self.min_val, dtype=torch.float32)
            return torch.full(
                (denorm_params[next(iter(denorm_params))].shape[0],),
                self.min_val,
                dtype=torch.float32,
                device=denorm_params[next(iter(denorm_params))].device,
            )

    def get_max(self, denorm_params):
        """
        Returns the maximum value of the variable, which can depend on other parameters.

        Args:
            denorm_params (dict): Dictionary of denormalized parameters.

        Returns:
            torch.Tensor: Maximum value of the variable.
        """
        if callable(self.max_val):
            return self.max_val(denorm_params)
        else:
            if denorm_params is None or not denorm_params:
                return torch.tensor(self.max_val, dtype=torch.float32)
            return torch.full(
                (denorm_params[next(iter(denorm_params))].shape[0],),
                self.max_val,
                dtype=torch.float32,
                device=denorm_params[next(iter(denorm_params))].device,
            )


class Constraint:
    """
    Represents a constraint in the optimization problem.
    """

    def __init__(self, expr_func, type="inequality"):
        """
        Initializes the Constraint.

        Args:
            expr_func (callable): Function that takes decision variables and parameters and returns a PyTorch expression.
            type (str): Type of constraint, either 'equality' or 'inequality'.
        """
        self.expr_func = expr_func
        assert type in [
            "equality",
            "inequality",
        ], "Constraint type must be 'equality' or 'inequality'"
        self.type = type
        self.n_constraints = None

    def get_constraints(self, decision_vars, params_dict):
        """
        Computes the constraint expression based on decision variables and parameters.

        Args:
            decision_vars (torch.Tensor): Decision variables.
            params_dict (dict): Dictionary of parameters {parameter_name: value}.

        Returns:
            torch.Tensor: Constraint expression.
        """
        expr = self.expr_func(decision_vars, params_dict)
        if isinstance(expr, list):
            expr = torch.stack(expr, dim=-1)
        elif isinstance(expr, torch.Tensor):
            expr = expr.reshape(expr.shape[0], -1)
        else:
            raise TypeError(
                "expr_func must return a torch.Tensor or a list of torch.Tensor"
            )

        if self.n_constraints is None:
            # Infer the number of constraints based on the expression's dimension
            self.n_constraints = expr.shape[-1]
        return expr


class OptimizationProblem:
    """
    Defines an optimization problem, including parameters, decision variables, cost function, and constraints.
    """

    def __init__(self, parameters, decision_variables, cost_function, constraints):
        """
        Initializes the OptimizationProblem.

        Args:
            parameters (list of Variable): List of parameter variables.
            decision_variables (list of Variable): List of decision variables.
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
        Counts the total number of equality and inequality constraints.

        Returns:
            tuple: (num_eq_constraints, num_ineq_constraints)
        """
        num_eq = 0
        num_ineq = 0
        # Create dummy parameters and decision variables
        dummy_params = {}
        for param in self.parameters:
            dummy_params[param.name] = torch.zeros(1)

        dummy_decision_vars = torch.zeros(1, len(self.decision_variables))

        for constraint in self.constraints:
            expr = constraint.get_constraints(dummy_decision_vars, dummy_params)
            if constraint.type == "equality":
                num_eq += expr.shape[-1]
            elif constraint.type == "inequality":
                num_ineq += expr.shape[-1]
        return num_eq, num_ineq

    def denormalize_parameters(self, norm_params):
        """
        Denormalizes the parameters from [-1, 1] to their original scale sequentially.

        Args:
            norm_params (torch.Tensor): Normalized parameters of shape (batch_size, num_parameters).

        Returns:
            dict: Dictionary of denormalized parameters {parameter_name: tensor_denormalized}.
        """
        denorm_params = {}
        for i, var in enumerate(self.parameters):
            denorm_min = var.get_min(denorm_params)
            denorm_max = var.get_max(denorm_params)
            denorm = (
                0.5 * (norm_params[:, i] + 1.0) * (denorm_max - denorm_min) + denorm_min
            )
            denorm_params[var.name] = denorm
        return denorm_params

    def normalize_parameters(self, params):
        """
        Normalizes the parameters from their original scale to the range [-1, 1], considering dependencies between parameters.

        Args:
            params (torch.Tensor): Unnormalized parameters of shape (batch_size, num_parameters).

        Returns:
            torch.Tensor: Normalized parameters of shape (batch_size, num_parameters).
        """
        norm_params = []
        denorm_params = (
            {}
        )  # Dictionary for the denormalized parameters processed so far
        for i, var in enumerate(self.parameters):
            # Extract the current denormalized parameter
            denorm_param = params[:, i]
            # Add the current denormalized parameter to the dictionary
            denorm_params[var.name] = denorm_param
            # Compute denormalized min and max values considering dependencies
            denorm_min = var.get_min(denorm_params)
            denorm_max = var.get_max(denorm_params)
            # Normalize the current parameter to the range [-1, 1]
            norm = 2.0 * (denorm_param - denorm_min) / (denorm_max - denorm_min) - 1.0
            norm_params.append(norm)
        # Combine all normalized parameters into a single tensor
        norm_params = torch.stack(norm_params, dim=1)
        return norm_params

    def denormalize_decision(self, norm_decisions, denorm_params):
        """
        Denormalizes the decision variables from [-1, 1] to their original scale.

        Args:
            norm_decisions (torch.Tensor): Normalized decision variables of shape (batch_size, num_decision_vars).
            denorm_params (dict): Dictionary of denormalized parameters {parameter_name: tensor_denormalized}.

        Returns:
            torch.Tensor: Denormalized decision variables of shape (batch_size, num_decision_vars).
        """
        denorm_decisions = []
        for i, var in enumerate(self.decision_variables):
            denorm_min = var.get_min(denorm_params)
            denorm_max = var.get_max(denorm_params)
            denorm = (
                0.5 * (norm_decisions[:, i] + 1.0) * (denorm_max - denorm_min)
                + denorm_min
            )
            denorm_decisions.append(denorm)
        denorm_decisions = torch.stack(denorm_decisions, dim=1)
        return denorm_decisions
