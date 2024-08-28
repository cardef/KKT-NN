import lightning as L
import pandas as pd
import numpy as np
from tqdm import tqdm
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
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
from torchmetrics.regression import R2Score
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
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.linear = nn.Linear(n, n).to(dtype=torch.float32, device=device)
        self.relu = nn.LeakyReLU()
        self.ln = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        identity = X
        y = self.relu(self.linear(self.dropout(self.ln(X))) + identity)

        return y
class Net(nn.Module):
    def softmax(self, input, t=1e-2):
        ex = torch.exp(input / t - torch.max(input / t, 1)[0].unsqueeze(1))
        sum = torch.sum(ex, axis=1).unsqueeze(1)
        return ex / sum

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(8),

            nn.Linear(8, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
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
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.device = device
        #self.mu = torch.tensor(mu.to_numpy(), dtype=torch.float32).to(device=self.device)
        #self.S = torch.tensor(S.to_numpy(), dtype=torch.float32).to(device=self.device)
        self.net = Net()
        self.tb_logger = SummaryWriter("loads/tb_logs")
        self.alpha = 0.9
        self.beta_p = 0.999
        self.tau = 1e-5
        self.n_iter = 0
        self.eps = 1e-4
        self.es= EarlyStopper(patience = 10000)
        self.plateau = False
        self.automatic_optimization = False
        self.optimizer = optim.NAdam(self.net.parameters(), lr=1E-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma =0.99999)
        self.coeffs = torch.ones(4, device=self.device, dtype=torch.float32)
        self.coeffs[0] = 1.0
        self.coeffs[1] = 1.0
        self.initial_losses = None
        self.previous_losses = None
        self.G_val_mult = torch.Tensor(
                [
                    [-1, 0, 0, 0],
                    [1, 0, 0 ,0],
                    [1, 0, 0, 0],
                    [0, 0, -1 ,0],
                    [0, 0, 1 ,0],
                    [1.5, 0, 1,0],
                    [1.5, 0, -1, 0],
                    [0, -1,0, 0],
                    [0, 1, 0 ,0],
                    [0, 1, 0, 0],
                    [0, 0, 0 ,-1],
                    [0, 0, 0 ,1],
                    [0, 2,0 ,1],
                    [0, 2, 0, -1],
                ]
            ).to(dtype=torch.float32, device=device)
        
        self.h_val_mult = torch.tensor(
                [
                    [-0.0000, 0.3000, 0.0400, 0.3000, 0.3000, 0.6000, 0.6000, -0.0000, 0.5000,
        0.1100, 0.5000, 0.5000, 1.2000, 1.2000]
                ],
            ).to(dtype=torch.float32, device=device)
        
        self.G_val = torch.Tensor(
                [
                    [-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, -1]
                ]
            ).to(dtype=torch.float32, device=device)
    def loss_grad_std_wn(self, loss, net):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "mps")
            grad_ = torch.zeros((0), dtype=torch.float32, device=device)
            for elem in torch.autograd.grad(loss, net.parameters(), retain_graph=True):
                grad_ = torch.cat((grad_, elem.view(-1)))
            mean = torch.mean(grad_)
            diffs = grad_ - mean
            var = torch.mean(torch.pow(diffs, 2.0))

            if var.item() == 0.0:
                return 0.0
            std = torch.pow(var, 0.5)
            zscores = diffs / std
            #kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
            return 1.0 / (torch.std(grad_) + torch.finfo(torch.float32).eps)

    def coeff_rel_improv(self, losses, prev_losses):
        with torch.no_grad():
            coeff_rel_improv = 4 * torch.exp(
                losses / (self.tau * prev_losses + torch.finfo(torch.float32).eps)
                - torch.max(
                    losses / (self.tau * prev_losses + torch.finfo(torch.float32).eps)
                )
            )

            coeff_rel_improv /= torch.sum(
                torch.exp(
                    losses / (self.tau * prev_losses + torch.finfo(torch.float32).eps)
                    - torch.max(
                        losses
                        / (self.tau * prev_losses + torch.finfo(torch.float32).eps)
                    )
                )
            )

            return coeff_rel_improv

    def kkt_loss(self, actions, P_pots, P_max, Q_max, P_plus, Q_plus, Q_minus, sol, lambd):

        beta = torch.bernoulli(torch.tensor(self.beta_p))
        G_val =self.G_val.repeat(actions.shape[0], 1, 1)
        
        tau1 = (Q_plus - Q_max)/(P_max - P_plus)
        tau2 = (Q_minus + Q_max)/(P_max - P_plus)

        rho1 = Q_max - tau1*P_plus
        rho2 = -Q_max - tau2*P_plus
        h_val =torch.stack((torch.zeros(512, device = self.device), P_max, P_pots, Q_max, Q_max, rho1, -rho2), 1)
        G_val[..., -2 , 0] = -tau1
        G_val[..., -1 , 0] = tau2
        #grad_g = torch.matmul(G_val.T, lambd.T).T
        grad_g = torch.bmm(lambd.unsqueeze(1), G_val).squeeze()
        grad_f = sol - actions
        #g_ineq = torch.matmul(G_val, sol.T).T - h_val
        g_ineq = torch.bmm(G_val, sol.unsqueeze(2)).squeeze() - h_val
        dual_feasibility = torch.relu(-lambd)
        stationarity = grad_f + grad_g
        complementary = lambd * g_ineq
        loss_stationarity = torch.norm(stationarity)
        loss_g_ineq = torch.norm(torch.relu(g_ineq))
        loss_complementary = torch.norm(complementary)
        loss_dual_feasibility = 0*torch.norm(dual_feasibility)
        loss_sparsity = torch.norm(sol, p=1)

    
        losses = torch.stack([loss_stationarity, loss_g_ineq, loss_complementary, loss_dual_feasibility])
        
        if self.initial_losses is None:
            self.initial_losses = losses

        if self.previous_losses is None:
            self.previous_losses = losses

        self.coeffs = self.alpha*(beta*self.coeffs + (1-beta)*self.coeff_rel_improv(losses, self.initial_losses)) + (1-self.alpha)*self.coeff_rel_improv(losses, self.previous_losses)
        
        self.previous_losses = losses

        return (
            self.coeffs@losses + 0.0*loss_sparsity,
            loss_stationarity,
            loss_g_ineq,
            loss_complementary,
            loss_dual_feasibility,
        )

    def training_step(self):
        
        #actions = X[..., :4]
        #P_pots = X[...,4:]
        #actions = torch.tensor([0.0, 0.0, -0.3, -0.5], device = self.device) + torch.rand((512, 4), device = self.device) * torch.tensor([0.3, 0.5, 0.6, 1.0], device = self.device)
        
        
        P_max = 0.8*torch.rand((512), device = self.device) +0.2
        Q_max = 0.8*torch.rand((512), device = self.device) +0.2
        Q_min = -Q_max

        P_plus = (P_max - 0.1)*torch.rand((512), device = self.device) + 0.1
        Q_plus = 0.1+torch.rand((512), device = self.device) * (Q_max - 0.1)
        Q_minus = -0.1 + torch.rand((512), device = self.device) * (Q_min + 0.1)
        P_pots = 0.0+torch.rand((512), device = self.device) * (P_max - 0.0)

        #P_max = torch.ones((512), device = self.device)*0.3
        #Q_max = torch.ones((512), device = self.device)*0.3
        Q_min = -Q_max

        #P_plus = torch.ones((512), device = self.device)*0.2
        #Q_plus = torch.ones((512), device = self.device)*0.15
        #Q_minus = torch.ones((512), device = self.device) * -0.15
        #Q_minus = -Q_plus
        actions = torch.tensor([1.0, 2.0], device = self.device) * torch.rand((512, 2), device = self.device) + torch.tensor([0.0, -1.0], device = self.device)

        
        def closure():
            sol, lambd = self.net(torch.cat([actions, P_pots.unsqueeze(1), P_max.unsqueeze(1), Q_max.unsqueeze(1), P_plus.unsqueeze(1), Q_plus.unsqueeze(1), Q_minus.unsqueeze(1)], 1))
            kkt_loss, stationarity, g_ineq, complementary, feasability = self.kkt_loss(actions.detach(), P_pots.detach(), P_max.detach(), Q_max.detach(), P_plus.detach(), Q_plus.detach(), Q_minus.detach(), sol, lambd
            )

            self.optimizer.zero_grad()
            kkt_loss.backward(retain_graph=True)
            self.tb_logger.add_scalars('run',
                {
                    "train_loss": kkt_loss,
                    "stationarity": stationarity,
                    "g_ineq": g_ineq,
                    "complementary": complementary,
                    "feasibility": feasability
                }, self.n_iter,
            )
            self.tb_logger.flush()
            self.n_iter += 1
            if self.es.early_stop(kkt_loss): self.plateau = True
            return kkt_loss
        def closure_lambd():
            kkt_loss, stationarity, g_eq, g_ineq, complementary = self.kkt_loss(
                X.detach(), sol.detach(), nu
            )

            self.optimizer_lambd.zero_grad()
            (-kkt_loss).backward(retain_graph=True)
            self.tb_logger.add_scalars('run',
                {
                    "train_loss": kkt_loss,
                    "stationarity": stationarity,
                    "g_eq": g_eq,
                    "g_ineq": g_ineq,
                    "complementary": complementary,
                }, self.n_iter,
            )
            self.tb_logger.flush()
            self.n_iter += 1
            return -kkt_loss
        self.optimizer.step(closure)
        self.scheduler.step()
        if self.plateau:
            if isinstance(self.optimizer, optim.LBFGS):
                print("a")
            else:
                input()
                self.optimizer = optim.LBFGS(self.net.parameters(), lr = 1e-6, history_size=1000, max_iter=1000, line_search_fn="strong_wolfe")
        #self.optimizer_lambd.step(closure_lambd)
    def validation_step(self, X, y):
        self.net.eval()
        X = X.to(self.device)
       
        y = y.to(self.device)
        with torch.no_grad():
            sol, _ = self.net(X)
            val_loss = nn.functional.mse_loss(sol, y)
            val_r2 = R2Score(2).to(self.device)(sol, y)
            #err =  torch.abs(-torch.log(X + sol).sum(1) + torch.log(X + y).sum(1))
            #err = 100*err/torch.abs(torch.log(X + y).sum(1))
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("R2/Val", val_r2,  self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1,0],  self.n_iter)
            #self.tb_logger.add_scalar("Sol/Err", err.mean(),  self.n_iter)
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
    train_set, val_set, _ = random_split(dataset, [0.1, 0.1, 0.8])
    train_loader = DataLoader(train_set, 2048, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, 2048, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, 512, shuffle=False, num_workers=5)
    model = KKT_NN()

    # res = trainer.test(model, test_loader)
    for epoch in tqdm(range(1000000)):
        
        model.training_step()

        for X, y in val_loader:
            model.validation_step(X, y)
    dump(res, open("res.pt", "wb"))
