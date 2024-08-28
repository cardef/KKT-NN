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


class Net(nn.Module):
    def softmax(self, input, t=1e-2):
        ex = torch.exp(input / t - torch.max(input / t, 1)[0].unsqueeze(1))
        sum = torch.sum(ex, axis=1).unsqueeze(1)
        return ex / sum

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Linear(64, 20),
        ).to(dtype=torch.float64, device=device)

    def forward(self, X):
        y = self.mlp(X)

        sol = torch.nn.functional.softmax(y, 1)

        return sol
class NetLambd(nn.Module):

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        ).to(dtype=torch.float64, device=device)

    def forward(self, X):
        y = self.mlp(X)

        #sol = torch.nn.functional.softmax(y[..., :-2], 1)
        lambd = torch.nn.functional.relu(y[..., [0]])
        nu = y[..., [-1]]

        return lambd, nu

class KKT_NN():

    def __init__(self, mu, S):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mu = torch.tensor(mu.to_numpy(), dtype=torch.float64).to(device=self.device)
        self.S = torch.tensor(S.to_numpy(), dtype=torch.float64).to(device=self.device)
        self.net = Net()
        self.net_lambd = NetLambd()
        self.tb_logger = SummaryWriter("KKT-NN/logs/portfolio")
        self.alpha = 0.9
        self.beta_p = 0.999
        self.tau = 1e-5
        self.n_iter = 0
        self.eps = 1e-4
        self.automatic_optimization = False
        self.optimizer = optim.AdamW(self.net.parameters(), lr=3e-4)
        self.optimizer_lambd = optim.AdamW(self.net_lambd.parameters(), lr=3e-4)
        self.coeffs = torch.ones(4, device=self.device, dtype=torch.float64)
        self.coeffs[0] = 1.0
        #self.coeffs[2] = 0.01
        #self.coeffs[1] = 0.01
        self.initial_losses = None
        self.previous_losses = None

    def loss_grad_std_wn(self, loss, net):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grad_ = torch.zeros((0), dtype=torch.float64, device=device)
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
            return 1.0 / (torch.std(grad_) + torch.finfo(torch.float64).eps)

    def coeff_rel_improv(self, losses, prev_losses):
        with torch.no_grad():
            coeff_rel_improv = 4 * torch.exp(
                losses / (self.tau * prev_losses + torch.finfo(torch.float64).eps)
                - torch.max(
                    losses / (self.tau * prev_losses + torch.finfo(torch.float64).eps)
                )
            )

            coeff_rel_improv /= torch.sum(
                torch.exp(
                    losses / (self.tau * prev_losses + torch.finfo(torch.float64).eps)
                    - torch.max(
                        losses
                        / (self.tau * prev_losses + torch.finfo(torch.float64).eps)
                    )
                )
            )

            return coeff_rel_improv

    def kkt_loss(self, x, sol, lambd, nu):

        beta = torch.bernoulli(torch.tensor(self.beta_p))

        grad_f = torch.matmul(self.S, sol.T).T
        g_ineq = -torch.matmul(sol, self.mu) + x.squeeze()
        g_eq = sol.sum(dim=1) - 1.0
        dual_feasibility = torch.relu(-lambd)
        stationarity = grad_f + lambd * self.mu + nu * torch.ones_like(sol)
        complementary = lambd * g_ineq
        loss_stationarity = torch.mean(torch.abs(stationarity))
        loss_g_eq = torch.mean(torch.relu(torch.abs(g_eq) - self.eps))
        loss_g_ineq = torch.mean(torch.relu((torch.abs(torch.relu(g_ineq) - self.eps))))
        loss_complementary = torch.mean(torch.relu(torch.abs(complementary) - self.eps))
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
                    + torch.finfo(torch.float64).eps
                )
            ) """
        if self.initial_losses is None:
            self.initial_losses = losses

        if self.previous_losses is None:
            self.previous_losses = losses

        #self.coeffs = self.alpha*(beta*self.coeffs + (1-beta)*self.coeff_rel_improv(losses, self.initial_losses)) + (1-self.alpha)*self.coeff_rel_improv(losses, self.previous_losses)
        #self.coeffs = self.coeffs / (self.coeffs[0] + torch.finfo(torch.float64).eps)
        self.previous_losses = losses

        print(self.coeffs)
        return (
            lagrangian.mean(),
            loss_stationarity,
            loss_g_eq,
            loss_g_ineq,
            loss_complementary,
        )

    def training_step(self, X, y):

        X = torch.rand(512).to(dtype=torch.float64, device=self.device) * 0.3
        

        def closure():
            sol = self.net(X.unsqueeze(1))
            lambd, nu = self.net_lambd(X.unsqueeze(1))
            kkt_loss, stationarity, g_eq, g_ineq, complementary = self.kkt_loss(
                X.detach(), sol, lambd.detach(), nu.detach()
            )

            self.optimizer.zero_grad()
            kkt_loss.backward(retain_graph=True)
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
            return kkt_loss
        def closure_lambd():
            sol = self.net(X.unsqueeze(1))
            lambd, nu = self.net_lambd(X.unsqueeze(1))
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
        self.optimizer_lambd.step(closure_lambd)
    def validation_step(self, X, y):
        self.net.eval()
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            print(X.shape, y.shape)
            sol = self.net(X.unsqueeze(1))
            val_loss = nn.functional.mse_loss(sol, y)
            val_r2 = R2Score(20).to(self.device)(sol, y)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("R2/Val", val_r2,  self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1,0],  self.n_iter)
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
        self.samples = load(open("KKT-NN/dataset_stock.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        y = np.array(y)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return X.astype(np.float64), y.astype(np.float64)


if __name__ == "__main__":
    df = pd.read_csv("KKT-NN/stock_prices.csv").drop("date", axis=1).iloc[:, :20]
    mu = mean_historical_return(df)
    S = CovarianceShrinkage(df).ledoit_wolf()
    dataset = Samples()
    train_set, val_set = random_split(dataset, [0.1, 0.9])
    train_loader = DataLoader(train_set, 512, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, 512, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, 512, shuffle=False, num_workers=5)
    model = KKT_NN(mu, S)

    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )
    logger = TensorBoardLogger("KKT-NN/tb_logs", name="my_model")
    # res = trainer.test(model, test_loader)
    for epoch in tqdm(range(1000000)):
        for X, y in train_loader:
            model.training_step(X, y)

        for X, y in val_loader:
            model.validation_step(X, y)
    dump(res, open("res.pt", "wb"))
