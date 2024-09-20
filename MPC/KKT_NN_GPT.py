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
        self.horizon = 2
        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Linear(64, self.horizon*5),
        )

    def forward(self, X):
        y = self.mlp(X)
        return (
            torch.sigmoid(y[..., :self.horizon]),
            torch.relu(y[..., self.horizon:self.horizon*2]),
            torch.relu(y[..., self.horizon*2:self.horizon*3]),
            torch.relu(y[..., self.horizon*3:self.horizon*4]),
            torch.relu(y[..., self.horizon*4:self.horizon*5]),
        )


class KKT_NN:
    def __init__(self):
        self.device = torch.device("cpu")
        self.net = Net().to(self.device)
        self.horizon = 2
        self.batch_size = 1
        self.n_iter = 0
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.net.parameters(), lr=3e-4, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=1.0)

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("MPC/runs/tb_logs/" + current_time)

    def cost(self, U, x_t, x_ref, a, b, q, r):
        x = x_t
        cost_value = 0
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x_next = a * x + b * U_t
            cost_value += q * (x_next - x_ref[:, t].squeeze()).pow(2) + r * U_t.pow(2)
            x = x_next.detach()
        return cost_value.sum()
        #return (((a*x_t + b*U) - x_ref)**2).sum() + r*((U**2).sum())

    def grad_cost(self, U, x_t, x_ref, a, b, q, r):
        grad = torch.zeros_like(U)
        x = x_t
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x_next = a * x + b * U_t
            grad[:, t] = 2 * b * q * (x_next - x_ref[:, t].squeeze()) + 2 * r * U_t
            x = x_next
        return grad

    def kkt_loss(self, U, lambda_, mu, nu, rho, x_t, x_ref, a, b, q, r):
        U_min = 0.0
        U_max = 1.0
        T_min = 0.0
        T_max = 1.0
        
        # Violazione della stazionarietà
        grad_J = self.grad_cost(U, x_t, x_ref, a, b, q, r)
        grad_J = torch.autograd.grad(self.cost(U, x_t, x_ref, a, b, q, r), U)[0]
        grad_ineq_control = lambda_ - mu
        grad_ineq_state = -b.unsqueeze(1) * nu + b.unsqueeze(1) * rho
        stationarity_violation = (grad_J + grad_ineq_control + grad_ineq_state).pow(2).sum()

        # Violazione della primal feasibility (controlli e stati)
        control_feasibility = (torch.max(U_min - U, torch.zeros_like(U)) + torch.max(U - U_max, torch.zeros_like(U))).pow(2).sum()

        x = x_t
        state_feasibility = 0
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x = a * x + b * U_t
            state_feasibility += (torch.max(T_min - x, torch.zeros_like(x)) + torch.max(x - T_max, torch.zeros_like(x))).pow(2).sum()

        # Violazione della complementarità
        comp_control = lambda_ * (U - U_min) + mu * (U_max - U)
        comp_state = 0
        x = x_t
        for t in range(self.horizon):
            U_t = U[:, t].squeeze()
            x_next = a * x + b * U_t
            nu_t = nu[:, t].squeeze()
            rho_t = rho[:, t].squeeze()
            comp_state += (nu_t * (T_min - x_next) + rho_t * (x_next - T_max)).pow(2).sum()
            x = x_next
        comp_loss = comp_control.pow(2).sum() + comp_state

        # Violazione della dual feasibility

        # Somma delle violazioni
        return stationarity_violation, control_feasibility,state_feasibility,  comp_loss

    def training_step(self):
        r = 0.3*torch.rand((self.batch_size), device=self.device) + 0.1
        a = 0.4*torch.rand((self.batch_size), device=self.device) + 0.6
        q = torch.ones(self.batch_size, device=self.device)
        b = 0.2*torch.rand((self.batch_size), device=self.device) + 0.1
        x_t = torch.rand((self.batch_size), device=self.device)
        x_ref = torch.rand((self.batch_size), device=self.device) 
        
        r = torch.ones((self.batch_size), device=self.device) * 0.1
        a = torch.ones((self.batch_size), device=self.device) * 0.9
        b = torch.ones((self.batch_size), device=self.device) * 0.1
        sol, lambda_, mu, nu, rho = self.net(torch.stack([x_t, x_ref], 1))
        loss_stationarity, loss_control, loss_state, loss_comp = self.kkt_loss(sol, lambda_, mu, nu, rho, x_t.detach(), x_ref.repeat(self.horizon, 1, ).T.detach(), a.detach(), b.detach(), q.detach(), r.detach())
        
        self.optimizer.zero_grad()
        torchjd.backward([loss_stationarity, loss_control, loss_state, loss_comp], self.net.parameters(), self.agg)
        #(loss_stationarity + loss_comp+loss_state).backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0e-5)
        self.optimizer.step()
        self.scheduler.step()

        self.tb_logger.add_scalar("Loss/Sum", loss_stationarity+loss_control+loss_state+loss_comp, self.n_iter)
        self.tb_logger.add_scalar("Loss/Control", loss_control, self.n_iter)
        self.tb_logger.add_scalar("Loss/State", loss_state, self.n_iter)
        self.tb_logger.add_scalar("Loss/Stat", loss_stationarity, self.n_iter)
        self.tb_logger.add_scalar("Loss/Comp", loss_comp, self.n_iter)
        self.tb_logger.flush()
        self.n_iter += 1

        return (loss_stationarity+loss_control+loss_state+loss_comp).item()

    def validation_step(self, X, y):
        self.net.eval()
        with torch.no_grad():
            X = X.to(self.device)
       
            y = y.to(self.device)
            sol, lambda_, mu, nu, rho = self.net(X)
            x_t = X[:, 0]
            x_ref = X[:, 1]
            #a = X[:, 2]
            #b = X[:, 3]
            #r = X[:, 4]

            a = torch.tensor(0.9)
            b = torch.tensor(0.1)
            r = torch.tensor(0.1)
            val_loss = nn.functional.l1_loss(sol, y)
            val_r2 = R2Score(self.horizon).to(self.device)(sol, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(sol, y)
            self.tb_logger.add_scalar("Val/Loss", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Val/R2", val_r2,  self.n_iter)
            self.tb_logger.add_scalar("Val/MAPE", val_mape,  self.n_iter)
            self.tb_logger.add_scalar("Loss/Val", val_loss, self.n_iter)
            self.tb_logger.add_scalar("Sol/True", y[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Sol/Pred", sol[-1,0],  self.n_iter)
            self.tb_logger.add_scalar("Lambd", lambda_.max(),  self.n_iter)
            self.tb_logger.add_scalar("Cost", self.cost(sol, x_t, x_ref.repeat(self.horizon, 1, ).T, a, b, torch.tensor(1.0), r),  self.n_iter)
            self.tb_logger.add_scalar("Cost/True", self.cost(y, x_t, x_ref.repeat(self.horizon, 1, ).T, a, b, torch.tensor(1.0), r),  self.n_iter)
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
            model.validation_step(X, y)
