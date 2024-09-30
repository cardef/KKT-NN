import lightning as L
import pandas as pd
import numpy as np
import torchjd
from torchjd.aggregation import UPGrad, MGDA, NashMTL, DualProj
from tqdm import tqdm
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from datetime import datetime
import torch
from torch import (
    nn,
    optim,
    sin,
    cos,
    mean,
    where,
    rand,
    Tensor,
    randperm,
    cat,
    full,
    bernoulli,
)
from torchmetrics.regression import R2Score, MeanAbsolutePercentageError
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from scipy.stats import qmc
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from pickle import load, dump

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
        device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        self.linear = nn.Linear(n, n).to(dtype=torch.float32, device=device)
        self.relu = nn.LeakyReLU()
        self.ln = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        identity = X
        #y = self.relu(self.linear(self.dropout(self.ln(X))) + identity)

        return self.relu(self.linear(X)+identity)
class Net(nn.Module):

    def __init__(self):
        super().__init__()
        device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        self.mlp = nn.Sequential(
            #nn.BatchNorm1d(8),

            nn.Linear(5, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 25),
        ).to(dtype=torch.float32, device=device)

    def forward(self, X):
        y = self.mlp(X)

        sol = torch.nn.functional.softmax(y[..., :-2], dim = 1)
        lambda_ = torch.nn.functional.relu(y[..., 5:10])
        mu = torch.nn.functional.relu(y[..., 10:15])
        nu = torch.nn.functional.relu(y[..., 15:20])
        rho = torch.nn.functional.relu(y[..., 20:25])
        return y[..., :5], lambda_, mu, nu, rho


class KKT_NN():

    def __init__(self):
        super().__init__()
        device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        self.device = device
        self.net = Net()
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("loads/tb_logs/" + current_time)
        self.horizon = 5
        self.n_iter = 0
        self.eps = 1e-4
        self.batch_size = 1024
        self.es= EarlyStopper(patience = 1000)
        self.plateau = False
        self.terminated = False
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma =0.9986)
        self.agg = UPGrad()

    def cost(U, x_t, x_ref, a, b, q, r):
        cost_value = 0
        x = x_t
        for t in range(self.horizon):
            U_t = U[:, t, :]  # controllo al passo t
            x = a.unsqueeze(1) * x + b.unsqueeze(1) * U_t  # aggiornamento dello stato
            cost_value += q.unsqueeze(1) * (x - x_ref[:, t, :]).pow(2) + r.unsqueeze(1) * U_t.pow(2)
        return cost_value.sum()

# Gradiente della funzione di costo rispetto a U
    def grad_cost(self, U, x_t, x_ref, a, b, q, r):
        grad = torch.zeros_like(U)
        x = x_t
        for t in range(self.horizon):
            U_t = U[:, t, :]
            x_next = a.unsqueeze(1) * x + b.unsqueeze(1) * U_t
            grad[:, t, :] = 2 * b.unsqueeze(1) * q.unsqueeze(1) * (x_next - x_ref[:, t, :]) + 2 * r.unsqueeze(1) * U_t
            x = x_next
        return grad

# Funzione obiettivo basata sulle condizioni KKT
    def kkt_loss(self, U, lambda_, mu, nu, rho, x_t, x_ref, a, b, q, r):

        U_min = torch.zeros(self.batch_size, 1)
        U_max = torch.ones(self.batch_size, 1)
        T_min = torch.zeros(self.batch_size, 1)
        T_max = torch.ones(self.batch_size, 1)
        T_min = T_min.unsqueeze(1).expand(self.batch_size, self.horizon, 1)
        T_max = T_max.unsqueeze(1).expand(self.batch_size, self.horizon, 1)
        # 1. Violazione della stazionarietà
        grad_J = self.grad_cost(U, x_t, x_ref, a, b, q, r)
        grad_ineq_control = lambda_ - mu
        grad_ineq_state = -b.unsqueeze(1).unsqueeze(2) * nu + b.unsqueeze(1).unsqueeze(2) * rho
        stationarity_violation = (grad_J + grad_ineq_control + grad_ineq_state).pow(2).sum()

        # 2. Violazione della primal feasibility (controlli e stati)
        control_feasibility = (torch.max(U_min.unsqueeze(1) - U, torch.zeros_like(U)) + torch.max(U - U_max.unsqueeze(1), torch.zeros_like(U))).pow(2).sum()
        
        x = x_t
        state_feasibility = 0
        for t in range(self.horizon):
            U_t = U[:, t, :]
            x = a.unsqueeze(1) * x + b.unsqueeze(1) * U_t
            state_feasibility += (torch.max(T_min - x, torch.zeros_like(x)) + torch.max(x - T_max, torch.zeros_like(x))).pow(2).sum()

        # 3. Violazione della complementarità
        comp_control = lambda_ * (U - U_min.unsqueeze(1)) + mu * (U_max.unsqueeze(1) - U)
        comp_state = 0
        x = x_t
        for t in range(self.horizon):
            U_t = U[:, t, :]
            x_next = a.unsqueeze(1) * x + b.unsqueeze(1) * U_t
            comp_state += (nu * (T_min - t_next) + rho * (t_next - T_max)).pow(2).sum()

            x = x_next
        comp_loss = comp_control.pow(2).sum() + comp_state

        # 4. Violazione della dual feasibility
        dual_feasibility = (torch.min(lambda_, torch.zeros_like(lambda_)).pow(2) + torch.min(mu, torch.zeros_like(mu)).pow(2) +
                            torch.min(nu, torch.zeros_like(nu)).pow(2) + torch.min(rho, torch.zeros_like(rho)).pow(2)).sum()

        # Somma delle violazioni
        return stationarity_violation + control_feasibility + state_feasibility + comp_loss + dual_feasibility



    def training_step(self):
        self.net.train()
        
        
        r = torch.rand((self.batch_size), device = self.device, dtype=torch.float32)
        a = torch.rand((self.batch_size), device = self.device, dtype=torch.float32)
        q = torch.ones(self.batch_size, device=self.device)
        b = torch.rand((self.batch_size), device = self.device, dtype=torch.float32)
        x_t = torch.rand((self.batch_size), device = self.device, dtype=torch.float32)
        x_ref = torch.rand((self.batch_size), device = self.device, dtype=torch.float32)

        
        
        def closure():
            sol, lambda_, mu, nu, rho = self.net(torch.stack([r, a, b, x_t, x_ref], 1))
            kkt_loss, stationarity, g_ineq, complementary = self.kkt_loss(sol.unsqueeze(-1), lambda_.unsqueeze(-1), mu.unsqueeze(-1), nu.unsqueeze(-1), rho.unsqueeze(-1), x_t.unsqueeze(-1), x_ref.repeat(self.horizon, 1, 1).permute(2,0,1), a, b, q, r
            )

            self.optimizer.zero_grad()
            #kkt_loss.backward()
            torchjd.backward([stationarity, g_ineq, complementary], self.net.parameters(), self.agg)
            #torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.tb_logger.add_scalars('run',
                {
                    "train_loss": kkt_loss,
                    "stationarity": stationarity,
                    "g_ineq": g_ineq,
                    "complementary": complementary,
                }, self.n_iter,
            )
            self.tb_logger.flush()
            self.n_iter += 1
            if self.es.early_stop(kkt_loss): self.plateau = True
            return kkt_loss
        self.optimizer.step(closure)
        self.scheduler.step()
        if self.plateau:
            if isinstance(self.optimizer, optim.LBFGS):
                self.terminated = True
            else:
                input()
                self.optimizer = optim.LBFGS(self.net.parameters(), lr = 1e-6, history_size=1000, max_iter=1000, line_search_fn="strong_wolfe")
        return self.terminated

    def validation_step(self, X, y):
        self.net.eval()
        with torch.no_grad():
            X = X.to(self.device)
       
            y = y.to(self.device)
            sol, lambd = self.net(X)
            val_loss = nn.functional.l1_loss(sol, y)
            val_r2 = R2Score(2).to(self.device)(sol, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(sol, y)
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2,  self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape,  self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Lambd", lambd.max(),  self.n_iter)
            self.tb_logger.flush()

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.mlp(X)
        test_mse = nn.functional.mse_loss(y_hat, y)
        print(y_hat.view(-1))
        test_r2 = R2Score().to(self.device)(y_hat.view(-1), y.view(-1))
        self.log("test_mse", test_mse)
        self.log("test_r2", test_r2)



class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("loads/dataset_loads.pkl", "rb"))
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
        return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    
    dataset = Samples()
    loader = DataLoader(dataset, 512, shuffle=False, num_workers=0)
    model = KKT_NN()
    terminated = False
    pbar = tqdm()
    while not terminated:
        terminated = model.training_step()

        pbar.update(1)
        for X, y in loader:
            model.validation_step(X, y)
        
    torch.save(model.net.state_dict(), "loads/kkt_nn.pt")
