from kinn.optimization_problem import OptimizationProblem, Variable, Constraint
from kinn.kinn import KINN
import torch

parameters = [
    Variable("action_P", 0.0, 1.0),
    Variable("action_Q", -1.0, 1.0),
    Variable("P_max", 0.2, 1.0),
    Variable("Q_max", 0.2, 1.0),
    Variable("P_pot", 0.0, lambda params: params["P_max"]),
    Variable(
        "P_plus", 0.1, lambda params: 0.9 * params["P_max"]
    ),  # Upper bound depends on P_max (second parameter)
    Variable("Q_plus", 0.1, lambda params: 0.9 * params["Q_max"]),
]

# Define Decision Variables
decision_variables = [
    Variable("P", 0.0, lambda params: params["P_pot"]),  # Normalized between [-1, 1]
    Variable("Q", lambda params: -params["Q_max"], lambda params: params["Q_max"]),
]


def cost_function(decision_vars, param):
    return torch.square(param["action_P"] - decision_vars[..., 0]) + torch.square(
        param["action_Q"] - decision_vars[..., 1]
    )


def p_constraints(decision_vars, param):
    P_max = param["P_max"]
    P_pot = param["P_pot"]

    return [
        -decision_vars[..., 0],
        decision_vars[..., 0] - P_max,
        decision_vars[..., 0] - P_pot,
    ]


def q_constraints(decision_vars, param):
    Q_max = param["Q_max"]

    return [
        -decision_vars[..., 1] - Q_max,
        decision_vars[..., 1] - Q_max,
    ]


def linear_constraints(decision_vars, param):
    P_max = param["P_max"]
    Q_max = param["Q_max"]
    P_plus = param["P_plus"]
    Q_plus = param["Q_plus"]
    tau_1 = (Q_plus - Q_max) / (P_max - P_plus)
    tau_2 = (-Q_plus + Q_max) / (P_max - P_plus)

    rho_1 = Q_max - tau_1 * P_plus
    rho_2 = -Q_max - tau_2 * P_plus

    return [
        decision_vars[..., 1] - tau_1 * decision_vars[..., 0] - rho_1,
        -decision_vars[..., 1] + tau_2 * decision_vars[..., 0] + rho_2,
    ]


constraints = [
    Constraint(expr_func=p_constraints, type="inequality"),
    Constraint(expr_func=q_constraints, type="inequality"),
    Constraint(expr_func=linear_constraints, type="inequality"),
]


problem = OptimizationProblem(
    parameters=parameters,
    decision_variables=decision_variables,
    cost_function=cost_function,
    constraints=constraints,
)

validation_filepath = "Projection/projection.pkl"


model = KINN(
    problem=problem,
    validation_filepath=validation_filepath,
    learning_rate=3e-4,
    early_stop_patience=2000,
    scheduler_patience=1000,
)


num_steps = 100000
batch_size = 1024
print("Starting training...")
model.train_model(
    num_steps=num_steps,
    batch_size=batch_size,
)


model.save_model("Projection/kkt_nn_projection.pth")
model.save_metrics("Projection/projection_metrics.csv")
model.save_losses("Projection/projection_losses.csv")
print("Model and metrics saved.")
