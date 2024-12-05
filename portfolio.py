from kinn.optimization_problem import OptimizationProblem, Variable, Constraint
from kinn.kinn import KINN
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import torch
import pandas as pd

df = pd.read_csv("Portfolio/stock_prices.csv").drop("date", axis=1)
mu = torch.tensor(mean_historical_return(df).to_numpy(dtype="float32"))
S = torch.tensor(CovarianceShrinkage(df).ledoit_wolf().to_numpy(dtype="float32"))

parameters = [
    Variable("max_tech", 0.0, 1.0),
    Variable("max_retail", 0.0, 1.0),
    Variable("max_fig", 0.0, 1.0),
    Variable("max_energy", 0.0, 1.0),
    Variable("utility_all", 0.0, 0.2),
    Variable("airline_all", 0.0, 0.2),
    Variable("rho", 1.0, 10.0),
]

decision_variables = [Variable(f"w_{i+1}", 0.0, 1.0) for i in range(len(mu))]


def cost_function(decision_variables, parameters):
    rho = parameters["rho"]
    cost_value = -(torch.dot(decision_variables, mu) - 0.5 * rho * (torch.dot(decision_variables, S @ decision_variables)))
    return cost_value





def equality_constraints(decision_variables, parameters):
    utility_all = parameters["utility_all"]
    airline_all = parameters["airline_all"]
    if utility_all.ndim == 0:
        utility_all = utility_all.unsqueeze(0)
        airline_all = airline_all.unsqueeze(0)
    return [decision_variables.sum(-1) - 1.0, decision_variables[..., 5] - utility_all, decision_variables[..., 11] - airline_all]

def inequality_constraints(decision_variables, parameters):
    max_tech = parameters['max_tech']
    max_retail = parameters['max_retail']
    max_fig = parameters['max_fig']
    max_energy = parameters['max_energy']

    tech = decision_variables[..., [0,1,2,3,4,6]]
    retail = decision_variables[..., [7, 12, 15, 19]]
    fig = decision_variables[..., [8, 16, 18]]
    energy = decision_variables[..., [13, 14]]

    return [tech.sum(-1) - max_tech, retail.sum(-1) - max_retail, fig.sum(-1) - max_fig, energy.sum(-1) - max_energy]
constraints = [
    Constraint(expr_func=equality_constraints, type="equality"),
    Constraint(expr_func=inequality_constraints, type="inequality"),
]

problem = OptimizationProblem(
    parameters=parameters,
    decision_variables=decision_variables,
    cost_function=cost_function,
    constraints=constraints,
)


validation_filepath = "Portfolio/portfolio.pkl"


model = KINN(
    problem=problem,
    validation_filepath=validation_filepath,
    num_embedding_residual_block=4,
    num_outputs_residual_block=3,
    hidden_dim=512,
    learning_rate=3e-4,
    early_stop_patience=2000,
    scheduler_patience=1000,
    device=torch.device("cpu"),
)


num_steps = 10000
batch_size = 1024
print("Starting training...")
model.train_model(
    num_steps=num_steps,
    batch_size=batch_size,
)


model.save_model("Portfolio/kkt_nn_portfolio.pth")
model.save_metrics("Portfolio/portfolio_metrics.csv")
model.save_losses("Portfolio/portfolio_losses.csv")
print("Model and metrics saved.")
