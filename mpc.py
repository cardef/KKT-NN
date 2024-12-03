from kinn.optimization_problem import OptimizationProblem, Variable, Constraint
from kinn.kinn import KINN
import torch
from copy import copy

horizon = 100

parameters = [
    Variable("a", 0.8, 1.0),
    Variable("b", 0.1, 1.0),
    Variable("r", 0.1, 1.0),
    Variable("T_0", 15.0, 30.0),
    Variable("T_ref", 20.0, 25.0),
]

decision_variables = [Variable(f"U_{i+1}", 0.0, 10.0) for i in range(horizon)]


def cost_function(decision_variables, parameters):
    T_0 = parameters["T_0"].clone()
    a = parameters["a"]
    b = parameters["b"]
    T_ref = parameters["T_ref"]
    r = parameters["r"]
    horizon = decision_variables.shape[-1]
    cost_value = 0.0
    for t in range(horizon):
        T_next = a * T_0 + b * decision_variables[..., t]
        cost_value += (T_next - T_ref).pow(2) + r * decision_variables[..., t].pow(2)
        T_0 = T_next
    return cost_value/horizon


def state_constraint(decision_variables, parameters):
    state_constraints = []
    T_0 = parameters["T_0"].clone()
    a = parameters["a"]
    b = parameters["b"]
    horizon = decision_variables.shape[-1]
    T_max = 300.0
    T_min = 0.0
    for t in range(horizon):
        T_next = a * T_0 + b * decision_variables[..., t]

        state_constraints.append(T_next - T_max)
        state_constraints.append(T_min - T_next)

        T_0 = T_next
    return torch.stack(state_constraints, -1)


def change_constraint(decision_variables, parameters):
    change = torch.abs(decision_variables[..., 1:] - decision_variables[..., :-1])
    return change - 1.0


constraints = [
    Constraint(expr_func=change_constraint, type="inequality"),
]

problem = OptimizationProblem(
    parameters=parameters,
    decision_variables=decision_variables,
    cost_function=cost_function,
    constraints=constraints,
)


validation_filepath = "MPC/mpc.pkl"


model = KINN(
    problem=problem,
    validation_filepath=validation_filepath,
    num_embedding_residual_block=4,
    num_outputs_residual_block=1,
    hidden_dim=512,
    learning_rate=3e-4,
    early_stop_patience=1000,
    scheduler_patience=500,
    device=torch.device("cpu"),
)


num_steps = 100000
batch_size = 1024
print("Starting training...")
model.train_model(
    num_steps=num_steps,
    batch_size=batch_size,
)


model.save_model("MPC/kkt_nn_mpc.pth")
model.save_metrics("MPC/mpc_metrics.csv")
model.save_losses("MPC/mpc_losses.csv")
print("Model and metrics saved.")
