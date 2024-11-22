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
    def constraint(decision_vars, param):
        P_max = param[..., 2]
        Q_max = param[..., 3]
        P_pot = param[..., 4]
        P_plus = param[..., 5]
        Q_plus = param[..., 6]
        tau_1 = (Q_plus - Q_max) / (P_max - P_plus)
        tau_2 = (-Q_plus + Q_max) / (P_max - P_plus)

        rho_1 = Q_max - tau_1 * P_plus
        rho_2 = -Q_max - tau_2 * P_plus

        # Inequality constraints for the optimization problem.
        """ G = torch.stack(
            [
                torch.stack(
                    [
                        torch.full((P_max.shape[0],), -1.0, device=tau_1.device),
                        torch.ones(P_max.shape[0], device=tau_1.device),
                        torch.ones(P_max.shape[0], device=tau_1.device),
                        torch.zeros(P_max.shape[0], device=tau_1.device),
                        torch.zeros(P_max.shape[0], device=tau_1.device),
                        -tau_1,
                        tau_2,
                    ],
                    dim=1,
                ),
                torch.tensor(
                    [0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0], device=tau_1.device
                ).repeat(P_max.shape[0], 1),
            ],
            dim=2,
        )
        h = torch.stack(
            (
                torch.zeros(P_max.shape[0], dtype=torch.float32),
                P_max,
                P_pot,
                Q_max,
                Q_max,
                rho_1,
                -rho_2,
            ),
            1,
        ) """

        # h = np.array([P_max, P_max, P_pot, Q_max, Q_max, rho_1, -rho_2])
        """ G = torch.Tensor(
            [[-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [-tau_1, 1], [tau_2, -1]]
        ).to(dtype=torch.float32, device=tau_1.device)

        h = torch.Tensor([P_max, P_max, P_pot, Q_max, Q_max, rho_1, -rho_2]).to(
            dtype=torch.float32, device=tau_1.device
        ) """
        if decision_vars.ndim == 42:
            return torch.bmm(G, decision_vars.unsqueeze(2)).squeeze() - h

        else:
            # return G @ decision_vars - h
            return torch.stack(
                [
                    -decision_vars[..., 0],
                    decision_vars[..., 0] - P_max,
                    decision_vars[..., 0] - P_pot,
                    -decision_vars[..., 1] - Q_max,
                    decision_vars[..., 1] - Q_max,
                    decision_vars[..., 1] - tau_1*decision_vars[..., 0] - rho_1,
                    -decision_vars[..., 1] + tau_1*decision_vars[..., 0] + rho_2,
                ], decision_vars[..., 0].ndim
            )

    # Create a list of constraints
    constraints = [
        Constraint(expr_func=constraint, type="inequality"),
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
    model = KKT_NN(problem=problem, validation_filepath=validation_filepath)

    # Generate the Validation Dataset if it does not exist
    if not os.path.exists(validation_filepath):
        print("Generating validation dataset with CVXPY...")
        model.generate_validation_dataset(num_samples=1000, solver=cp.ECOS)
        print("Validation dataset generated.")

    # Load the Validation Dataset
    validation_loader = model.load_validation_dataset(
        filepath=validation_filepath, batch_size=512
    )

    # Train the Model
    num_steps = 10000  # Total number of training steps
    batch_size = 512  # Batch size for training
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

    # Example Prediction with Single Parameters
    new_params = [0.5, 0.6, 0.7, 0.0, 0.5]  # Unnormalized parameters
    optimal_solution = get_optimal_solution(model_loaded, new_params)
    print(f"\nParameters: {new_params}")
    print(f"Optimal Solution: {optimal_solution}")

    # Example Prediction with Batch Parameters
    new_params_batch = [
        [0.5, 0.6, 0.7, 0.0, 0.5],
        [0.3, 0.4, 0.5, 0.0, 0.3],
        [0.8, 0.9, 1.0, 0.0, 0.7],
    ]
    optimal_solutions_batch = get_optimal_solution(model_loaded, new_params_batch)
    print(f"\nBatch Parameters: {new_params_batch}")
    print(f"Optimal Solutions for Batch:\n{optimal_solutions_batch}")


if __name__ == "__main__":
    main()
