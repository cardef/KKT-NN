# user_example.py

import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from src.kkt_nn import Variable, Constraint, OptimizationProblem, KKT_NN
import cvxpy as cp


def main():
    # Define Parameter Variables
    parameters = [
        Variable("action_P", 0.0, 1.0),
        Variable("action_Q", -1.0, 1.0),
        Variable("P_max", 0.2, 1.0),
        Variable("Q_max", 0.2, 1.0),
        Variable("P_pots", 0.0, lambda params: params["P_max"]),
        Variable(
            "P_plus", 0.1, lambda params: 0.9 * params["P_max"]
        ),  # Upper bound depends on P_max (second parameter)
        Variable("Q_plus", 0.1, lambda params: 0.9 * params["Q_max"]),
    ]

    # Define Decision Variables
    decision_variables = [
        Variable("P", 0.0, lambda params: params["P_pots"]),  # Normalized between [-1, 1]
        Variable("Q", lambda params: -params["Q_max"], lambda params: params["Q_max"]),
    ]

    # Define the Cost Function
    def cost_function(decision_vars, param):
        """
        Cost function: Minimize x1^2 + x2^2.

        Args:
            decision_vars (torch.Tensor): Decision variables from PyTorch.
            param (torch.Tensor): Parameters of the problem.

        Returns:
            torch.Tensor: Computed cost.
        """
        return torch.square(param[..., 0:2] - decision_vars[..., :]).sum(-1)

    # Define Constraints using PyTorch Functions
    
    
    def p_constraints(decision_vars, param):
        P_max = param[..., 2]
        P_pot = param[..., 4]

        
        return [
                -decision_vars[..., 0],
                decision_vars[..., 0] - P_max,
                decision_vars[..., 0] - P_pot,
            ]
    
    def q_constraints(decision_vars, param):
        Q_max = param[..., 3]

        
        return [
                -decision_vars[..., 1] - Q_max,
                decision_vars[..., 1] - Q_max,
            ]
    def linear_constraints(decision_vars, param):
        P_max = param[..., 2]
        Q_max = param[..., 3]
        P_pot = param[..., 4]
        P_plus = param[..., 5]
        Q_plus = param[..., 6]
        tau_1 = (Q_plus - Q_max) / (P_max - P_plus)
        tau_2 = (-Q_plus + Q_max) / (P_max - P_plus)

        rho_1 = Q_max - tau_1 * P_plus
        rho_2 = -Q_max - tau_2 * P_plus

        
        return [
                decision_vars[..., 1] - tau_1*decision_vars[..., 0] - rho_1,
                -decision_vars[..., 1] + tau_2*decision_vars[..., 0] + rho_2,
            ]

    # Create a list of constraints
    constraints = [
        Constraint(expr_func=p_constraints, type="inequality"),
        Constraint(expr_func=q_constraints, type="inequality"),
        Constraint(expr_func=linear_constraints, type="inequality"),
    ]

    # Instantiate the Optimization Problem
    problem = OptimizationProblem(
        parameters=parameters,
        decision_variables=decision_variables,
        cost_function=cost_function,
        constraints=constraints,
    )

    # Path to the Validation Dataset
    validation_filepath = "projection.pkl"  # Change the name if preferred

    # Initialize the KKT Neural Network Model
    model = KKT_NN(problem=problem, validation_filepath=validation_filepath, learning_rate=3e-4)

    # Generate the Validation Dataset if it does not exist

    # Load the Validation Dataset
    validation_loader = model.load_validation_dataset(
        filepath=validation_filepath, batch_size=512
    )

    # Train the Model
    num_steps = 10000  # Total number of training steps
    batch_size = 1000  # Batch size for training
    print("Starting training...")
    model.train_model(
        num_steps=num_steps, batch_size=batch_size, validation_loader=validation_loader
    )

    # Save the Trained Model and Metrics
    model.save_model("kkt_nn_model_cvxpy.pth")
    model.save_metrics("training_metrics_cvxpy.csv")
    print("Model and metrics saved.")

    # Load the Trained Model for Prediction
    model_loaded = KKT_NN(problem=problem, validation_filepath=validation_filepath)
    model_loaded.load_model("kkt_nn_model_cvxpy.pth")
    print("Model loaded for prediction.")

    # Prediction Function
    def get_optimal_solution(model, input_params):
        """
        Retrieves the optimal solution given unnormalized input parameters.

        Args:
            model (KKT_NN): Trained KKT neural network model.
            input_params (list or np.ndarray or torch.Tensor): Unnormalized input parameters.

        Returns:
            np.ndarray: Optimal solution.
        """
        solution = model.predict(input_params)
        return solution


if __name__ == "__main__":
    main()
