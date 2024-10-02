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
        self.ln = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        identity = X
        return self.relu(self.linear(X) + identity)


class SolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 5
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, self.horizon),
            nn.Sigmoid(),
        )

    def forward(self, r, x_ref):

        X = torch.stack([r, x_ref], -1)
        y = self.mlp(X)

        return y


class LambdaNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 5
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, self.horizon * 4),
            nn.ReLU(),
        )

    def forward(self, r, x_ref):
        X = torch.stack([r, x_ref], -1)
        y = self.mlp(X)

        return y


class KKT_NN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.solnet = SolNet().to(self.device)
        self.lambdanet = LambdaNet().to(self.device)
        self.sobol_eng = torch.quasirandom.SobolEngine(2, scramble=True, seed=42)
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
        self.tb_logger = SummaryWriter("MPC/runs/tb_logs/" + current_time)

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
        r_unnorm = 0.1*r + 0.1
        x_ref_unnorm = 0.1 * x_ref + 0.7
        
        U = self.solnet(r, x_ref)

        lambda_ = self.lambdanet(r, x_ref)
        feasibility = self.constraints(a, b, x_0, U)
        complementarity = lambda_ * feasibility
        # grad_L = torch.autograd.grad(self.cost(a, b, r, x_0, x_ref, U), U, grad_outputs=torch.ones_like(U), is_grads_batched=True)[0]
        grad_L = vmap(grad(self.lagrangian, argnums=5), in_dims=(0, 0, 0, 0, 0, 0, 0))(
            a, b, r_unnorm, x_0, x_ref_unnorm, U, lambda_
        )

        loss_stationarity = torch.square(grad_L).sum(1).mean(0)
        loss_feasibility = torch.square(torch.relu(feasibility)).sum(1).mean(0)
        loss_complementarity = torch.square(complementarity).sum(1).mean(0)
        return loss_stationarity, loss_feasibility, loss_complementarity

    def training_step(self):
        sampled_batch = self.sobol_eng.draw(self.batch_size)
        a = torch.ones(self.batch_size) * 0.9
        b = torch.ones(self.batch_size) * 0.1
        r = torch.ones(self.batch_size) * 0.1
        x_0 = torch.ones(self.batch_size) * 0.2
        r = sampled_batch[..., 0]
        x_ref = sampled_batch[..., 1]
        # U, lambda_, mu = self.net(torch.stack([a, b, r, x_0, x_ref, t], 1))

        loss_stationarity, loss_feasibility, loss_complementarity = self.kkt_loss(a, b, r, x_0, x_ref)

        self.optimizer.zero_grad()
        self.lambda_optimizer.zero_grad()
        # torchjd.backward([loss_stationarity, loss_control, loss_state, loss_comp], self.net.parameters(), self.agg)
        (loss_stationarity+ loss_feasibility+loss_complementarity).backward()
        # torch.nn.utils.clip_grad_norm_(self.solnet.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(self.lambdanet.parameters(), 1.0)
        self.optimizer.step()
        self.lambda_optimizer.step()
        self.scheduler.step()
        self.lambda_scheduler.step()
        self.tb_logger.add_scalar(
            "Loss/Sum",
            loss_stationarity+ loss_feasibility+loss_complementarity,
            self.n_iter,
        )
        self.tb_logger.add_scalar("Loss/Stat", loss_stationarity, self.n_iter)
        self.tb_logger.add_scalar("Loss/Feas", loss_feasibility, self.n_iter)
        self.tb_logger.add_scalar("Loss/Comp", loss_complementarity, self.n_iter)
        self.tb_logger.flush()
        self.n_iter += 1

        return (loss_stationarity+ loss_feasibility+loss_complementarity).item()

    def validation_step_fake(self):
        self.solnet.eval()
        self.lambdanet.eval()
        with torch.no_grad():
            U = self.solnet(torch.ones(X.shape[0]).unsqueeze(1) * 100.0)
        val_loss = nn.functional.l1_loss(
            U,
            torch.tensor(([0.87088543, 0.69774885, 0.54322603, 0.38843083, 0.21444385]))
            .unsqueeze(0)
            .repeat(U.shape[0], 1),
        )
        self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
        self.tb_logger.add_scalar("Val/U1", U[..., 0].mean(), self.n_iter)
        self.tb_logger.flush()

    def validation_step(self, r, x_ref, y):
        self.solnet.eval()
        with torch.no_grad():
            x_ref = x_ref.to(self.device)
            y = y.to(self.device)
            U = self.solnet(r, x_ref)
            val_loss = nn.functional.l1_loss(U, y)
            val_r2 = R2Score(self.horizon).to(self.device)(U, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(U, y)
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2, self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape, self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[0, 0], self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", U[0, 0], self.n_iter)
            # self.tb_logger.add_scalar("Lambd", lambda_.max(), self.n_iter)
            self.tb_logger.flush()


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("MPC/mpc.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r, x_ref, y = self.samples[idx]
        r = np.array(r)
        x_ref = np.array(x_ref)
        y = np.array(y)
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return (
            r.astype(np.float32),
            x_ref.astype(np.float32),
            y.astype(np.float32),
        )


if __name__ == "__main__":
    dataset = Samples()
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    model = KKT_NN()
    pbar = tqdm()

    for epoch in range(100000):
        pbar.set_description(f"Epoch {epoch+1}")
        model.training_step()
        for r, x_ref, y in loader:
            model.validation_step(r, x_ref, y)
