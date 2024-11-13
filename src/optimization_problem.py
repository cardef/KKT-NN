# optimization_problem.py
import torch
from kkt_optimizer import KKTOptimizer

class OptimizationProblem:
    def __init__(self):
        self.variables = {}
        self.variable_order = []
        self.cost_function = None
        self.inequality_constraints = []
        self.equality_constraints = []
        self.variable_dependencies = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def add_variable(self, name, dim, bounds, dependencies=None):
        self.variables[name] = {
            'dim': dim,
            'bounds': bounds,
            'dependencies': dependencies or []
        }
        self.variable_order.append(name)
        if dependencies:
            self.variable_dependencies[name] = dependencies

    def set_cost_function(self, cost_function):
        self.cost_function = cost_function

    def add_inequality_constraint(self, constraint_function):
        self.inequality_constraints.append(constraint_function)

    def add_equality_constraint(self, constraint_function):
        self.equality_constraints.append(constraint_function)

    def solve(self, epochs=100, batch_size=64, learning_rate=1e-3):
        optimizer = KKTOptimizer(
            problem=self,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epochs=epochs
        )
        optimizer.optimize()
        self.optimizer = optimizer  # Save the optimizer for future use

    def get_solution(self, **input_params):
        return self.optimizer.get_solution(**input_params)
