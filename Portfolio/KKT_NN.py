import lightning as L
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchjd
from torchjd.aggregation import UPGrad, MGDA, NashMTL, DualProj
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
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.linear = nn.Linear(n, n).to(dtype=torch.float32, device=device)
        self.relu = nn.LeakyReLU()
        self.ln = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.0)

    def forward(self, X):
        identity = X
        y = self.relu(self.linear(self.dropout(self.ln(X))) + identity)

        return self.relu(self.linear(X) + identity)
class Net(nn.Module):
    def softmax(self, input, t=1e-2):
        ex = torch.exp(input / t - torch.max(input / t, 1)[0].unsqueeze(1))
        sum = torch.sum(ex, axis=1).unsqueeze(1)
        return ex / sum

    def __init__(self):
        super().__init__()
        device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        self.mlp = nn.Sequential(
            #nn.LayerNorm(1),

            nn.Linear(421, 1024),
            nn.LeakyReLU(),
            ResidualBlock(1024),
            ResidualBlock(1024),
            nn.Linear(1024, 22),
        ).to(dtype=torch.float32, device=device)

    def forward(self, S, mu, X):
        y = self.mlp(torch.cat((S, mu, X), 1))
        #y = self.mlp(X)
        sol = torch.nn.functional.softmax(y[..., :-2], 1)
        lambd = torch.nn.functional.relu(y[..., [-2]])
        nu = y[..., [-1]]

        return sol, lambd, nu


class KKT_NN():

    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu" if torch.cuda.is_available() else "mps")
        #self.mu = torch.tensor(mu.to_numpy(), dtype=torch.float32).to(device=self.device)
        #self.S = torch.tensor(S.to_numpy(), dtype=torch.float32).to(device=self.device)
        self.net = Net()
        self.tb_logger = SummaryWriter("portfolio/tb_logs")
        self.alpha = 0.9
        self.beta_p = 0.999
        self.tau = 1e-5
        self.n_iter = 0
        self.eps = 1e-4
        self.es= EarlyStopper(patience = 10000)
        self.plateau = False
        self.automatic_optimization = False
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-4)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma =0.9999)
        self.agg = UPGrad()
        self.coeffs = torch.ones(4, device=self.device, dtype=torch.float32)
        self.initial_losses = None
        self.previous_losses = None

    def loss_grad_std_wn(self, loss, net):
        with torch.no_grad():
            device = torch.device("cpu" if torch.cuda.is_available() else "mps")
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

    def kkt_loss(self, S, mu, x, sol, lambd, nu):

        beta = torch.bernoulli(torch.tensor(self.beta_p))

        grad_f = torch.bmm(S.view(128, 20, 20), sol.unsqueeze(-1)).squeeze()
        g_ineq = -torch.bmm(mu.unsqueeze(1), sol.unsqueeze(-1)).squeeze() + x.squeeze()
        g_eq = sol.sum(dim=1) - 1.0
        dual_feasibility = torch.relu(-lambd)
        stationarity = grad_f + lambd * mu + nu * torch.ones_like(sol)
        complementary = lambd.squeeze() * g_ineq
        loss_stationarity = torch.norm(stationarity, dim=1).mean()
        loss_g_eq = torch.norm(g_eq, dim=0).mean()
        loss_g_ineq = (torch.relu(g_ineq)).mean()
        loss_complementary = complementary.mean()
        # loss_dual_feasibility = torch.norm(dual_feasibility)**2
        loss_sparsity = torch.norm(sol, p=1)
        lagrangian  = torch.bmm(sol.unsqueeze(1), grad_f.unsqueeze(2)).squeeze() + nu.squeeze()*g_eq + lambd.squeeze()*g_ineq
        losses = torch.stack(
            [
                loss_stationarity,
                loss_g_eq,
                loss_g_ineq,
                loss_complementary,
            ]
        )
        """ for i in range(1, losses.shape[0]):
            self.coeffs[i] = self.alpha * self.coeffs[i] + (1 - self.alpha) * (
                self.loss_grad_std_wn(losses[i], self.net)
                / (
                    self.loss_grad_std_wn(losses[0], self.net)
                    + torch.finfo(torch.float32).eps
                )
            ) """
        if self.initial_losses is None:
            self.initial_losses = losses

        if self.previous_losses is None:
            self.previous_losses = losses

        #self.coeffs = self.alpha*(beta*self.coeffs + (1-beta)*self.coeff_rel_improv(losses, self.initial_losses)) + (1-self.alpha)*self.coeff_rel_improv(losses, self.previous_losses)
        #self.coeffs = self.coeffs / (self.coeffs[0] + torch.finfo(torch.float32).eps)
        self.previous_losses = losses
        #if loss_g_ineq < 1e-4:
            #self.coeffs[2] = 0.0
        #if loss_g_ineq >= 5e-3:
        #    self.coeffs[2] = 1.0
        return (
            self.coeffs@losses,
            loss_stationarity,
            loss_g_eq,
            loss_g_ineq,
            loss_complementary,
        )

    def training_step(self, S, mu, X, y):

        self.net.train()
        A = torch.rand((128, 20, 20)).to(dtype=torch.float32, device=self.device)
        #S= torch.tensor(S.to_numpy()).flatten().repeat(128,1).to(device = self.device, dtype= torch.float32)
        #mu= torch.tensor(mu.to_numpy()).repeat(128,1).to(device = self.device, dtype= torch.float32)
        S = torch.bmm(torch.transpose(A, 1, 2), A).to(dtype=torch.float32, device=self.device).flatten(1)
        S = (S - S.min(dim=0)[0]) / (S.max(dim = 0)[0] - S.min(dim = 0)[0])
        mu = 2.0 * torch.rand(128, 20).to(dtype=torch.float32, device=self.device) - 1.0
        X = (torch.min(mu, 1)[0] + 0.1)+ torch.rand(128).to(dtype=torch.float32, device=self.device) * ((torch.max(mu,1)[0] - 0.1) - torch.min(mu, 1)[0] + 0.1)
        X = X.unsqueeze(-1)
        def closure():
            sol, lambd, nu = self.net(S, mu, X)
            kkt_loss, stationarity, g_eq, g_ineq, complementary = self.kkt_loss(
                S.detach(), mu.detach(), X.detach(), sol, lambd, nu
            )

            self.optimizer.zero_grad()
            torchjd.backward([stationarity, g_eq, g_ineq, complementary], self.net.parameters(), self.agg)
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
            if self.es.early_stop(kkt_loss): self.plateau = True
            return kkt_loss
        def closure_lambd():
            kkt_loss, stationarity, g_eq, g_ineq, complementary = self.kkt_loss(
                X.detach(), sol.detach(), lambd, nu
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
    def validation_step(self, S, mu, X, y):
        self.net.eval()
        X = X.to(self.device).unsqueeze(-1)
        S= torch.tensor(S.to_numpy()).flatten().repeat(X.shape[0],1).to(device = self.device, dtype= torch.float32)
        mu= torch.tensor(mu.to_numpy()).repeat(X.shape[0],1).to(device = self.device, dtype= torch.float32)
       
        y = y.to(self.device)
        with torch.no_grad():
            sol, _, _ = self.net(S.flatten(1), mu , X)
            val_loss = nn.functional.mse_loss(sol, y)
            val_r2 = R2Score(20).to(self.device)(sol, y)
            err = torch.bmm(sol.unsqueeze(1), torch.bmm(S.view(sol.shape[0], 20, 20), sol.unsqueeze(-1))).squeeze() - torch.bmm(y.unsqueeze(1), torch.bmm(S.view(sol.shape[0], 20, 20), y.unsqueeze(-1))).squeeze() 
            err = 100*err/torch.bmm(y.unsqueeze(1), torch.bmm(S.view(sol.shape[0], 20, 20), y.unsqueeze(-1))).squeeze() 
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("R2/Val", val_r2,  self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Err", err.mean(),  self.n_iter)
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
        self.samples = load(open("portfolio/dataset_stock.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        y = np.array(y)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    df = pd.read_csv("portfolio/stock_prices.csv").drop("date", axis=1).iloc[:, :20]
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    dataset = Samples()
    train_set, val_set = random_split(dataset, [0.1, 0.9])
    train_loader = DataLoader(train_set, 512, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, 512, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, 512, shuffle=False, num_workers=5)
    model = KKT_NN()

    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )
    logger = TensorBoardLogger("portfolio/tb_logs", name="my_model")
    # res = trainer.test(model, test_loader)
    for epoch in tqdm(range(1000000)):
        for X, y in train_loader:
            model.training_step(S, mu, X, y)

        for X, y in val_loader:
            model.validation_step(S, mu, X, y)
    dump(res, open("res.pt", "wb"))