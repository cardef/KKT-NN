import torch
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.func import grad, vmap, jacrev
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score, MeanAbsolutePercentageError, MeanSquaredError
import torchjd
from torchjd.aggregation import UPGrad, MGDA, NashMTL, DualProj
from datetime import datetime
from tqdm import tqdm
from pickle import load

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
class ResidualBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.relu = nn.LeakyReLU()
        self.ln = nn.LayerNorm(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        identity = X
        return self.relu(self.linear(X) + identity)


class KINN(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(5, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 512)
        )
        self.sol = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, horizon),
            nn.Sigmoid()
        )

        self.lambda_net= nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, horizon*4),
            nn.Softplus(beta=5),
        )
    def forward(self, X):
        embedding = self.shared(X)
        sol = self.sol(embedding)
        lambda_ = self.lambda_net(embedding)
        return sol, lambda_ 


class KKT_NN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.horizon = 10
        self.kinn = KINN(self.horizon).to(self.device)
        self.sobol_eng = torch.quasirandom.SobolEngine(5, scramble=True, seed=42,)
        self.batch_size = 64
        self.n_iter = 0
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.kinn.parameters(), lr=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
        self.terminated = False

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("MPC/runs/tb_logs/" + current_time)
        self.loss_stationarity = []
        self.loss_feasibility = []
        self.loss_complementarity = []
        self.r2 = []
        self.rmse = []
        self.mape = []
        self.mae = []
    def cost(self, a, b, r, x_0, x_ref, U):
        x = x_0
        cost_value = 0
        for t in range(self.horizon):
            U_t = U[..., t].squeeze()
            x_next = a * x + b * U_t
            cost_value += (x_next - x_ref).pow(2) + r * U_t.pow(2)
            x = x_next
        return cost_value

    def constraints(self, a, b, x_0, U):
        U_min = 0.0
        U_max = 1.0
        T_min = 0.0
        T_max = 1.0

        state_constraints = []
        x = x_0
        for t in range(self.horizon):
            U_t = U[..., t].squeeze()
            x_next = a * x + b * U_t

            state_constraints.append(x_next - T_max)
            state_constraints.append(T_min - x_next)

            x = x_next
        return torch.cat([U - U_max, U_min - U, torch.stack(state_constraints, -1)], -1)
    def lagrangian(self, a, b, r, x_0, x_ref, U, lambda_):

        if lambda_.ndim == 2:
            return self.cost(a, b, r, x_0, x_ref, U) + torch.bmm(lambda_.unsqueeze(1), self.constraints(a, b, x_0, U).unsqueeze(2)).squeeze()
        else:
            return self.cost(a, b, r, x_0, x_ref, U)+lambda_@self.constraints(a, b, x_0, U)
    def kkt_loss(self, a, b, r, x_0, x_ref):

        # Violazione della stazionarietà
        a_unnorm = 0.5*0.3*(a +1)+ 0.7
        b_unnorm = 0.5*0.4*(b +1) + 0.1
        r_unnorm = 0.5*1.0*(r +1) + 0.0
        x_0_unnorm = 0.5*1.0*(x_0 +1) + 0.0
        x_ref_unnorm = 0.5*1.0 * (x_ref +1) + 0.0
        
        U, lambda_ = self.kinn(torch.stack([a,b,r, x_0, x_ref], 1))
        feasibility = self.constraints(a_unnorm, b_unnorm, x_0_unnorm, U)
        complementarity = lambda_ * feasibility
        # grad_L = torch.autograd.grad(self.cost(a, b, r, x_0, x_ref, U), U, grad_outputs=torch.ones_like(U), is_grads_batched=True)[0]
        grad_L = vmap(grad(self.lagrangian, argnums=5), in_dims=(0, 0, 0, 0, 0, 0, 0))(
            a_unnorm, b_unnorm, r_unnorm, x_0_unnorm, x_ref_unnorm, U, lambda_
        )

        loss_stationarity = torch.square(grad_L).sum(1)
        loss_feasibility = torch.square(torch.relu(feasibility)).sum(1)
        loss_complementarity = torch.square(complementarity).sum(1)
        return loss_stationarity, loss_feasibility, loss_complementarity

    def training_step(self):
        self.solnet.train()
        self.lambdanet.train()
        sampled_batch = self.sobol_eng.draw(self.batch_size).to(self.device)
        a = torch.ones(self.batch_size, device=self.device) * 0.9
        b = torch.ones(self.batch_size, device=self.device) * 0.1
        r = torch.ones(self.batch_size, device=self.device) * 0.1
        x_0 = torch.ones(self.batch_size, device=self.device) * 0.2
        a = sampled_batch[..., 0]
        b = sampled_batch[..., 1]
        r = sampled_batch[..., 2]
        x_0 = sampled_batch[..., 3]
        x_ref = sampled_batch[..., 4]
        # U, lambda_, mu = self.net(torch.stack([a, b, r, x_0, x_ref, t], 1))
        def closure():
            self.optimizer.zero_grad()
            loss_stationarity, loss_feasibility, loss_complementarity = self.kkt_loss(a, b, r, x_0, x_ref)
            loss = (loss_stationarity + loss_feasibility+loss_complementarity).mean()
            loss.backward()
            self.loss_stationarity.append(loss_stationarity.mean())
            self.loss_feasibility.append(loss_feasibility.mean())
            self.loss_complementarity.append(loss_complementarity.mean())
            self.tb_logger.add_scalar(
            "Loss/Sum",
            loss,
            self.n_iter,
            )
            self.tb_logger.add_scalar("Loss/Stat", loss_stationarity.mean(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Feas", loss_feasibility.mean(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Comp", loss_complementarity.mean(), self.n_iter)
            self.tb_logger.flush()
            self.n_iter += 1

            if self.es.early_stop(loss):
                if isinstance(self.optimizer, optim.Adam) :
                    print("LBFGS")
                    self.es=EarlyStopper(patience=1000)
                    self.optimizer = optim.LBFGS(self.kinn.parameters(), lr=1e-2)
                else:
                    self.terminated = True
            #self.scheduler.step(loss)
            return loss
        
        self.optimizer.step(closure)
        
        
        return self.terminated


    def validation_step(self, X, y):
        self.solnet.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            U, lambda_ = self.kinn(X)
            val_loss = nn.functional.l1_loss(U, y)
            val_r2 = R2Score(self.horizon).to(self.device)(U, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(U, y)
            rmse = MeanSquaredError(squared=False).to(self.device)(U, y)

            self.r2.append(val_r2.item())
            self.mape.append(val_mape.item())
            self.mae.append(val_loss.item())
            self.rmse.append(rmse.item())
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2, self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape, self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[0, -1], self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", U[0, -1], self.n_iter)
            self.tb_logger.add_scalar("Lambd", lambda_.max(), self.n_iter)
            self.tb_logger.flush()


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("MPC/mpc.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        X = np.array(X)
        y = np.array(y)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return (
            X.astype(np.float32),
            y.astype(np.float32),
        )


if __name__ == "__main__":
    dataset = Samples()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model = KKT_NN()
    pbar = tqdm()

    epoch = 0
    while True:
        pbar.set_description(f"Epoch {epoch+1}")
        terminated = model.training_step()
        for X, y in loader:
            model.validation_step(X, y)
        epoch += 1

        if terminated:
            log = pd.DataFrame({'steps': [i for i in range(1,model.n_iter +1 )], 'stat': model.loss_stationarity, 'feas': model.loss_feasibility, 'compl': model.loss_complementarity}) 
            break

    log = pd.DataFrame({'Step': [i for i in range(1,model.n_iter +1 )], 'Stationarity': model.loss_stationarity, 'Feasibility': model.loss_feasibility, 'Complementarity': model.loss_complementarity}) 
    metrics = pd.DataFrame({'Step': [i for i in range(1,model.n_iter +1 )], 'R2': model.r2, 'MAPE': model.mape, 'MAE': model.mae, 'RMSE': model.rmse}) 
    log.to_csv("Projection/log.csv", index=False)
    metrics.to_csv("Projection/metrics.csv", index=False)
    torch.save(model.kinn.state_dict(), "MPC/kinn.pt")