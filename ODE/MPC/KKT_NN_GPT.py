import torch
import numpy as np
from torch import nn, optim
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


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.horizon = 5
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, self.horizon * 3),
        )

    def forward(self, X):
        y = self.mlp(X)

        y_hat = torch.zeros_like(y) + (1 - torch.exp(-X[:,[-1]])).repeat(1, y.shape[1]) * y


        """ return (
            torch.sigmoid(y_hat[..., : self.horizon]),
            torch.relu(y_hat[..., self.horizon : self.horizon * 2]),
            torch.relu(y_hat[..., self.horizon * 2 : self.horizon * 3]),
        ) """

        return y_hat[..., : self.horizon], y_hat[..., self.horizon : self.horizon * 2], y_hat[..., self.horizon * 2 : self.horizon * 3]


class KKT_NN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.net = Net().to(self.device)
        self.horizon = 5
        self.batch_size = 64
        self.n_iter = 0
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-5, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("ODE/MPC/runs/tb_logs/" + current_time)

    def cost(self, a, b, r, x_0, x_ref, U):
        x = x_0
        cost_value = 0
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x_next = a * x + b * U_t
            cost_value += (x_next - x_ref).pow(2) + r * U_t.pow(2)
            x = x_next
        return cost_value.sum()

    def control_constraints(self, U):
        U_min = 0.0
        U_max = 1.0

        return (
            torch.max(U_min - U, torch.zeros_like(U))
            + torch.max(U - U_max, torch.zeros_like(U))
        ).pow(2)

    def state_constraints(self, a, b, x_0, U):
        T_min = 0.0
        T_max = 1.0

        x = x_0
        state_feasibility = torch.zeros_like(U)
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x = a * x + b * U_t
            state_feasibility[..., t] = (
                torch.max(T_min - x, torch.zeros_like(x))
                + torch.max(x - T_max, torch.zeros_like(x))
            ).pow(2)
        return state_feasibility

    def kkt_loss(self, a, b, r, x_0, x_ref, t, U, lambda_, mu):

        # Violazione della stazionarietà

        control_feasibility = self.control_constraints(U)
        state_feasibility = self.state_constraints(a, b, x_0, U)

        grad_L = torch.autograd.grad(self.cost(a, b, r, x_0, x_ref, U), U)[0]
        jacob_control = torch.zeros((self.batch_size, self.horizon, self.horizon))

        for i in range(self.horizon):
            jacob_control[:, i, :] = torch.autograd.grad(
                control_feasibility[:, i],
                U,
                grad_outputs=torch.ones(self.batch_size),
                create_graph=True,
            )[0]

        jacob_state = torch.zeros((self.batch_size, self.horizon, self.horizon))

        for i in range(self.horizon):
            jacob_state[:, i, :] = torch.autograd.grad(
                state_feasibility[:, i],
                U,
                grad_outputs=torch.ones(self.batch_size),
                create_graph=True,
            )[0]

        dx = (
            -(grad_L
            + torch.bmm(
                torch.transpose(jacob_control, 1, 2), control_feasibility.unsqueeze(2)
            ).squeeze()
            + torch.bmm(
                torch.transpose(jacob_state, 1, 2), state_feasibility.unsqueeze(2)
            ).squeeze())
        )

        dlambda = 0.5 * control_feasibility
        dmu = 0.5 * state_feasibility

        loss = 0

        for i in range(self.horizon):
            loss += (
                torch.mean(
                    (
                        dx[:, i]
                        - torch.autograd.grad(
                            U[:, i], t, grad_outputs=torch.ones(self.batch_size), create_graph=True
                        )[0]
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        dlambda[:, i]
                        - torch.autograd.grad(
                            lambda_[:, i], t, grad_outputs=torch.ones(self.batch_size), create_graph=True
                        )[0]
                    )
                    ** 2
                )
                + torch.mean(
                    (
                        dmu[:, i]
                        - torch.autograd.grad(
                            mu[:, i], t, grad_outputs=torch.ones(self.batch_size), create_graph=True
                        )[0]
                    )
                    ** 2
                )
            )
        return loss

    def training_step(self):

        a = 0.4 * torch.rand((self.batch_size), device=self.device) + 0.6
        b = 0.2 * torch.rand((self.batch_size), device=self.device) + 0.1
        r = 0.3 * torch.rand((self.batch_size), device=self.device) + 0.1
        x_0 = torch.rand((self.batch_size), device=self.device)
        x_ref = torch.rand((self.batch_size), device=self.device)

        a = 0.9 * torch.ones((self.batch_size), device=self.device)
        b = 0.1 * torch.ones((self.batch_size), device=self.device)
        r = 0.1 * torch.ones((self.batch_size), device=self.device)
        x_0 = 0.3*torch.ones((self.batch_size), device=self.device)
        x_ref = 0.6*torch.ones((self.batch_size), device=self.device)
        t = torch.rand((self.batch_size), device=self.device) * 100.0
        t.requires_grad_(True)
        #U, lambda_, mu = self.net(torch.stack([a, b, r, x_0, x_ref, t], 1))
        U, lambda_, mu = self.net(t.unsqueeze(1))
        loss = self.kkt_loss(a, b, r, x_0, x_ref, t, U, lambda_, mu)

        self.optimizer.zero_grad()
        # torchjd.backward([loss_stationarity, loss_control, loss_state, loss_comp], self.net.parameters(), self.agg)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0e-5)
        self.optimizer.step()
        self.scheduler.step()
        
        """ self.tb_logger.add_scalar(
            "Loss/Sum",
            loss_stationarity + loss_control + loss_state + loss_comp,
            self.n_iter,
        )
        self.tb_logger.add_scalar("Loss/Control", loss_control, self.n_iter)
        self.tb_logger.add_scalar("Loss/State", loss_state, self.n_iter)
        self.tb_logger.add_scalar("Loss/Stat", loss_stationarity, self.n_iter)
        self.tb_logger.add_scalar("Loss/Comp", loss_comp, self.n_iter) """
        self.tb_logger.flush()
        self.n_iter += 1

        return loss.item()
    def validation_step_fake(self):
        self.net.eval()
        with torch.no_grad():
            U, lambda_, mu = self.net(torch.ones(X.shape[0]).unsqueeze(1)*100.0)
        val_loss = nn.functional.l1_loss(U, torch.tensor(([0.87088543, 0.69774885, 0.54322603, 0.38843083, 0.21444385])).unsqueeze(0).repeat(U.shape[0], 1))
        self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
        self.tb_logger.add_scalar("Val/U1", U[...,0].mean(), self.n_iter)
        self.tb_logger.flush()
    def validation_step(self, X, y):
        self.net.eval()
        with torch.no_grad():
            X = X.to(self.device)

            y = y.to(self.device)
            sol, lambda_, mu = self.net(torch.cat([X, torch.ones(X.shape[0]).unsqueeze(1)*100.0], 1))
            val_loss = nn.functional.l1_loss(sol, y)
            val_r2 = R2Score(self.horizon).to(self.device)(sol, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(sol, y)
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2, self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape, self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1, 0], self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1, 0], self.n_iter)
            self.tb_logger.add_scalar("Lambd", lambda_.max(), self.n_iter)
            self.tb_logger.flush()


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("ODE/MPC/mpc.pkl", "rb"))
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
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    model = KKT_NN()
    pbar = tqdm()

    for epoch in range(10000):
        pbar.set_description(f"Epoch {epoch+1}")
        model.training_step()
        for X, y in loader:
            model.validation_step_fake()
