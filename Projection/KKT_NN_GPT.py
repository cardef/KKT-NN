import torch
import numpy as np
from torch import nn, optim
from torch.func import grad, vmap, jacrev
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score, MeanAbsolutePercentageError
import torchjd
from torchjd.aggregation import UPGrad, MGDA, NashMTL, DualProj
from datetime import datetime
from tqdm import tqdm
from pickle import load


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


class SolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(7, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 2)
        )

    def forward(self, X):

        y = self.mlp(X)
        y[..., 0] = torch.sigmoid(y[...,0])
        y[..., 1] = torch.tanh(y[...,1])
        return y


class LambdaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 5
        self.mlp = nn.Sequential(
            nn.Linear(7, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 7),
            nn.ReLU(),
        )

    def forward(self, X):
        y = self.mlp(X)

        return  y


class KKT_NN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.solnet = SolNet().to(self.device)
        self.lambdanet = LambdaNet().to(self.device)
        self.sobol_eng = torch.quasirandom.SobolEngine(7, scramble=True, seed=42,)
        self.G = torch.Tensor(
                [
                    [-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, -1]
                ]
            ).to(dtype=torch.float32, device=self.device)
        self.horizon = 5
        self.batch_size = 64
        self.n_iter = 0
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.solnet.parameters(), lr=1e-5)
        self.lambda_optimizer = optim.Adam(self.lambdanet.parameters(), lr=1e-5)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99999)
        self.lambda_scheduler = optim.lr_scheduler.ExponentialLR(
            self.lambda_optimizer, gamma=0.99999
        )

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("Projection/runs/tb_logs/" + current_time)

    def cost(self, actions, sol):
        return torch.square(actions-sol).sum(-1)

    def constraints(self, G, h, sol):

        if sol.ndim == 2:
            return torch.bmm(G, sol.unsqueeze(2)).squeeze() - h
        
        else:
            return G@sol - h
    def lagrangian(self, actions, G, h, sol,lambda_):

        if lambda_.ndim == 2:
            return self.cost(actions, sol) + torch.bmm(lambda_.unsqueeze(1), self.constraints(G, h, sol).unsqueeze(2)).squeeze()
        else:
            return self.cost(actions, sol)+lambda_@self.constraints(G, h, sol)
    def kkt_loss(self, actions, P_pots, P_max, Q_max, P_plus, Q_plus,):

        # Violazione della stazionarietà
        G =self.G.repeat(actions.shape[0], 1, 1)
       
        tau1 = (Q_plus - Q_max)/(P_max - P_plus)
        tau2 = (-Q_plus + Q_max)/(P_max - P_plus)

        rho1 = Q_max - tau1*P_plus
        rho2 = -Q_max - tau2*P_plus
        h =torch.stack((torch.zeros(self.batch_size, device = self.device, dtype=torch.float32), P_max, P_pots, Q_max, Q_max, rho1, -rho2), 1)
        G[..., -2 , 0] = -tau1
        G[..., -1 , 0] = tau2
        
        sol = self.solnet(torch.stack([actions[...,0], actions[...,1], P_pots, P_max, Q_max, P_plus, Q_plus], 1))

        lambda_ = self.lambdanet(torch.stack([actions[...,0], actions[...,1], P_pots, P_max, Q_max, P_plus, Q_plus], 1))
        feasibility = self.constraints(G, h, sol)
        complementarity = lambda_ * feasibility
        # grad_L = torch.autograd.grad(self.cost(a, b, r, x_0, x_ref, U), U, grad_outputs=torch.ones_like(U), is_grads_batched=True)[0]
        grad_L = vmap(grad(self.lagrangian, argnums=3), in_dims=(0, 0, 0, 0, 0))(
            actions, G, h, sol,lambda_
        )

        loss_stationarity = torch.square(grad_L).sum(1)
        loss_feasibility = torch.square(torch.relu(feasibility)).sum(1)
        loss_complementarity = torch.square(complementarity).sum(1)
        return loss_stationarity, loss_feasibility, loss_complementarity

    def training_step(self):
        self.solnet.train()
        self.lambdanet.train()
        sampled_batch = self.sobol_eng.draw(self.batch_size).to(self.device)
        actions = torch.tensor([1.0, 2.0], device = self.device, dtype=torch.float32) * sampled_batch[..., :2]  + torch.tensor([0.0, -1.0], device = self.device, dtype=torch.float32)
        P_max = 0.8*sampled_batch[..., 2] + 0.2
        Q_max =  0.8*sampled_batch[..., 3] + 0.2
        P_plus = (0.9*P_max - 0.1)*sampled_batch[..., 4] + 0.1
        Q_plus = (0.9*Q_max - 0.1)*sampled_batch[..., 5] + 0.1
        P_pots = P_max*sampled_batch[..., 6]
        # U, lambda_, mu = self.net(torch.stack([a, b, r, x_0, x_ref, t], 1))

        loss_stationarity, loss_feasibility, loss_complementarity = self.kkt_loss(actions, P_pots, P_max, Q_max, P_plus, Q_plus)
        loss = (loss_stationarity + loss_feasibility+loss_complementarity).mean()
        self.optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        # torchjd.backward([loss_stationarity, loss_control, loss_state, loss_comp], self.net.parameters(), self.agg)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.solnet.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(self.lambdanet.parameters(), 1.0)
        self.optimizer.step()
        self.lambda_optimizer.step()
        self.scheduler.step()
        self.lambda_scheduler.step()
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

        return loss.item()

    def validation_step(self, X, y):
        self.solnet.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)
            sol = self.solnet(X)
            lambda_ = self.lambdanet(X)
            val_loss = nn.functional.l1_loss(sol, y)
            val_r2 = R2Score(2).to(self.device)(sol, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(sol, y)
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2, self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape, self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[0, -1], self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[0, -1], self.n_iter)
            self.tb_logger.add_scalar("Lambd", lambda_.max(), self.n_iter)
            self.tb_logger.flush()


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("Projection/projection.pkl", "rb"))
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
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    model = KKT_NN()
    pbar = tqdm()

    for epoch in range(10000):
        pbar.set_description(f"Epoch {epoch+1}")
        model.training_step()
        for X, y in loader:
            model.validation_step(X, y)
    torch.save(model.solnet.state_dict(), "MPC/solnet.pt")
    torch.save(model.lambdanet.state_dict(), "MPC/lambdanet.pt")