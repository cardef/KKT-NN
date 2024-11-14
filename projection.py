# projection.py

from src.optimization_problem import OptimizationProblem
import torch
from src.decorators import count_constraints  # Ensure the path is correct


def calculate_tau_rho(variables):
    """
    Calculates tau1, tau2, rho1, and rho2 based on the provided variables.

    Args:
        variables (dict): Dictionary of denormalized variables.

    Returns:
        tuple: (tau1, tau2, rho1, rho2) as tensors of shape [batch_size].
    """
    # Squeeze to convert [64, 1] to [64]
    P_max = variables['P_max'].squeeze(-1)       # Shape: [64]
    Q_max = variables['Q_max'].squeeze(-1)       # Shape: [64]
    P_plus = variables['P_plus'].squeeze(-1)     # Shape: [64]
    Q_plus = variables['Q_plus'].squeeze(-1)     # Shape: [64]
    
    # Prevent division by zero
    denominator = (P_max - P_plus)
    denominator = torch.where(denominator == 0, torch.ones_like(denominator) * 1e-6, denominator)
    
    tau1 = (Q_plus - Q_max) / denominator       # Shape: [64]
    tau2 = (-Q_plus + Q_max) / denominator      # Shape: [64]
    rho1 = Q_max - tau1 * P_plus                # Shape: [64]
    rho2 = -Q_max - tau2 * P_plus               # Shape: [64]
    return tau1, tau2, rho1, rho2


@count_constraints(num_constraints=7)
def inequality_constraints(variables):
    """
    Defines the inequality constraints for the optimization problem.

    Args:
        variables (dict): Dictionary of denormalized variables.

    Returns:
        torch.Tensor: Tensor of shape [batch_size, 7] representing the constraints for each sample.
    """
    # Access solution components correctly
    P = variables['solution'][0]  # Shape: [64]
    Q = variables['solution'][1]  # Shape: [64]

    # Access other variables, ensuring they are scalars and squeeze extra dimensions
    P_max = variables['P_max'].squeeze(-1)       # Shape: [64]
    Q_max = variables['Q_max'].squeeze(-1)       # Shape: [64]
    P_pots = variables['P_pots'].squeeze(-1)     # Shape: [64]

    # Calculate tau and rho values
    tau1, tau2, rho1, rho2 = calculate_tau_rho(variables)  # Each is [64]

    # Compute each constraint as a scalar
    constraint1 = -P                  # Shape: [64]
    constraint2 = P - P_max           # Shape: [64]
    constraint3 = P - P_pots          # Shape: [64]
    constraint4 = -Q                  # Shape: [64]
    constraint5 = Q - Q_max           # Shape: [64]
    constraint6 = -tau1 * P + Q - rho1 # Shape: [64]
    constraint7 = tau2 * P + Q - rho2  # Shape: [64]

    # Stack all constraints into a [64, 7] tensor
    constraints = torch.stack([
        constraint1,
        constraint2,
        constraint3,
        constraint4,
        constraint5,
        constraint6,
        constraint7
    ], dim=-1)  # Shape: [64, 7]

    # Debugging: Print shapes of constraints
    print("Constraint Shapes:")
    for i, constraint in enumerate([constraint1, constraint2, constraint3, constraint4, constraint5, constraint6, constraint7], 1):
        print(f"Constraint {i}: {constraint.shape}")

    return constraints  # Shape: [64, 7]


def main():
    # Define the problem
    problem = OptimizationProblem()

    # Add independent variables
    problem.add_variable('actions', dim=2, bounds=([0.0, -1.0], [1.0, 1.0]))
    problem.add_variable('P_max', dim=1, bounds=(0.2, 1.0))
    problem.add_variable('Q_max', dim=1, bounds=(0.2, 1.0))

    # Add dependent variables with bounds depending on other variables
    problem.add_variable(
        'P_plus', dim=1,
        bounds=(
            0.1,
            lambda variables: 0.9 * variables['P_max'] - 0.1
        )
    )

    problem.add_variable(
        'Q_plus', dim=1,
        bounds=(
            0.1,
            lambda variables: 0.9 * variables['Q_max'] - 0.1
        )
    )

    problem.add_variable(
        'P_pots', dim=1,
        bounds=(
            0.0,
            lambda variables: variables['P_max']
        )
    )

    # Add the decision variable 'solution'
    problem.add_variable('solution', dim=2, bounds=([0.0, 0.0], [1.0, 1.0]))

    # Define the cost function to minimize the difference between solution and actions
    def cost_function(variables):
        actions = variables['actions']
        solution = variables['solution']
        return torch.sum((solution - actions) ** 2, dim=-1)  # Shape: [batch_size]

    problem.set_cost_function(cost_function)

    # Add inequality constraints to the problem
    problem.add_inequality_constraint(inequality_constraints)

    # Optionally, define and add equality constraints
    @count_constraints(num_constraints=1)
    def equality_constraints(variables):
        """
        Enforces that the sum of the solution variables equals 1.

        Args:
            variables (dict): Dictionary of denormalized variables.

        Returns:
            torch.Tensor: Tensor of shape [batch_size, 1] representing the equality constraint for each sample.
        """
        solution = variables['solution']  # Shape: [64, 2]
        constraint = torch.sum(solution, dim=1, keepdim=True) - 1  # Shape: [64, 1]
        return constraint  # Shape: [64, 1]

    # Add equality constraints to the problem
    # Uncomment the line below if equality constraints are needed
    # problem.add_equality_constraint(equality_constraints)

    # Solve the optimization problem
    problem.solve(epochs=100, batch_size=64, learning_rate=1e-3)

    # Define input parameters for independent variables
    input_params = {
        'actions': [0.5, 0.2],
        'P_max': [0.8],
        'Q_max': [0.6]
        # Dependent variables are automatically handled
    }

    # Obtain the optimized solution
    solution = problem.get_solution(**input_params)
    print("Optimized Solution:", solution)


if __name__ == "__main__":
    main()
