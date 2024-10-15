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

            nn.Linear(7, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 9),
        ).to(dtype=torch.float32, device=device)

    def forward(self, X):
        y = self.mlp(X)

        sol = torch.nn.functional.softmax(y[..., :-2], dim = 1)
        lambd = torch.nn.functional.relu(y[..., 2:])

        return y[..., :2], lambd


class KKT_NN():

    def __init__(self):
        super().__init__()
        device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        self.device = device
        self.net = Net()
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("loads/tb_logs/" + current_time)
        self.alpha = 0.9999
        self.beta_p = 0.9999
        self.tau = 1e-1
        self.n_iter = 0
        self.eps = 1e-4
        self.es= EarlyStopper(patience = 1000)
        self.plateau = False
        self.terminated = False
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma =0.9986)
        self.agg = UPGrad()
        self.coeffs = torch.ones(3, device=self.device, dtype=torch.float32)
        self.initial_losses = None
        self.previous_losses = None
        
        self.G_val = torch.Tensor(
                [
                    [-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, -1]
                ]
            ).to(dtype=torch.float32, device=device)




    def kkt_loss(self, actions, P_pots, P_max, Q_max, P_plus, Q_plus, sol, lambd):

        beta = torch.bernoulli(torch.tensor(self.beta_p))
        G_val =self.G_val.repeat(actions.shape[0], 1, 1)
       
        tau1 = (Q_plus - Q_max)/(P_max - P_plus)
        tau2 = (-Q_plus + Q_max)/(P_max - P_plus)

        rho1 = Q_max - tau1*P_plus
        rho2 = -Q_max - tau2*P_plus
        h_val =torch.stack((torch.zeros(1024, device = self.device, dtype=torch.float32), P_max, P_pots, Q_max, Q_max, rho1, -rho2), 1)
        G_val[..., -2 , 0] = -tau1
        G_val[..., -1 , 0] = tau2
        grad_g = torch.bmm(lambd.unsqueeze(1), G_val).squeeze()
        grad_f = sol - actions
        g_ineq = torch.bmm(G_val, sol.unsqueeze(2)).squeeze() - h_val
        stationarity = grad_f + grad_g
        complementary = lambd * g_ineq
        loss_stationarity = torch.norm(stationarity, p=2, dim=1).mean()
        loss_g_ineq = torch.norm(torch.relu(g_ineq), p=2, dim=1).mean()
        loss_complementary = torch.norm(complementary, p=2, dim=1).mean()
        loss_sparsity = torch.norm(sol, p=1)

        
    
        losses = torch.stack([loss_stationarity, loss_g_ineq, loss_complementary])
        
       

        return (
            self.coeffs@losses + 0.0*loss_sparsity,
            loss_stationarity,
            loss_g_ineq,
            loss_complementary,
        )

    def training_step(self):
        self.net.train()
        
        
        P_max = 0.8*torch.rand((1024), device = self.device, dtype=torch.float32) +0.2
        Q_max = 0.8*torch.rand((1024), device = self.device, dtype=torch.float32) +0.2

        P_plus = (0.9*P_max - 0.1)*torch.rand((1024), device = self.device, dtype=torch.float32) + 0.1
        Q_plus = 0.1+torch.rand((1024), device = self.device, dtype=torch.float32) * (0.9*Q_max - 0.1)
        P_pots = 0.0+torch.rand((1024), device = self.device, dtype=torch.float32) * (P_max - 0.0)

        
        actions = torch.tensor([1.0, 2.0], device = self.device, dtype=torch.float32) * torch.rand((1024, 2), device = self.device, dtype=torch.float32) + torch.tensor([0.0, -1.0], device = self.device, dtype=torch.float32)

        
        def closure():
            sol, lambd = self.net(torch.cat([actions, P_pots.unsqueeze(1), P_max.unsqueeze(1), Q_max.unsqueeze(1), P_plus.unsqueeze(1), Q_plus.unsqueeze(1)], 1))
            kkt_loss, stationarity, g_ineq, complementary = self.kkt_loss(actions.detach(), P_pots.detach(), P_max.detach(), Q_max.detach(), P_plus.detach(), Q_plus.detach(), sol, lambd
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
        self.samples = load(open("loads/dataset_loads_comp_norm.pkl", "rb"))
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
