import torch
import random
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

random.seed(42)     # python random generator
np.random.seed(42)  # numpy random generator
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

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
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(7, 512),
            nn.LeakyReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 512)
        )
        self.sol = nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 2),
        )

        self.lambda_net= nn.Sequential(
            ResidualBlock(512),
            ResidualBlock(512),
            nn.Linear(512, 7),
            nn.Softplus(beta=5),
        )
    def forward(self, X):
        embedding = self.shared(X)
        sol = self.sol(embedding)
        sol[..., 0] = torch.sigmoid(sol[..., 0])
        sol[..., 1] = torch.tanh(sol[..., 1])
        lambda_ = self.lambda_net(embedding)
        return sol, lambda_ 
class KKT_NN:
    def __init__(self):
        self.device = 'cpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.kinn = KINN().to(self.device)
        self.sobol_eng = torch.quasirandom.SobolEngine(7, scramble=True, seed=42)
        self.G = torch.Tensor(
                [
                    [-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, -1]
                ]
            ).to(dtype=torch.float32, device=self.device)
        self.batch_size = 512
        self.n_iter = 0
        self.agg = UPGrad()
        self.es = EarlyStopper(patience=1000)
        self.optimizer = optim.Adam(self.kinn.parameters(), lr=3e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience= 1000, factor=0.1)
        self.terminated = False
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("Projection/runs/tb_logs/" + current_time)
        self.loss_stationarity = []
        self.loss_feasibility = []
        self.loss_complementarity = []
        self.r2 = []
        self.rmse = []
        self.mape = []
        self.mae = []

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

        actions_unnormed = torch.tensor([0.5, 1.0], device = self.device, dtype=torch.float32) * (actions + torch.tensor([1.0, 0.0], device = self.device, dtype=torch.float32)) 
        P_max_unnormed = 0.5*0.8*(P_max +1.0) + 0.2
        Q_max_unnormed =  0.5*0.8*(Q_max +1.0) + 0.2
        P_plus_unnormed = 0.5*(0.9*P_max_unnormed - 0.1)*(P_plus +1.0) + 0.1
        Q_plus_unnormed = 0.5*(0.9*Q_max_unnormed - 0.1)*(Q_plus +1.0) + 0.1
        P_pots_unnormed = 0.5*(P_pots +1.0) * P_max_unnormed

        # Violazione della stazionarietà
        G =self.G.repeat(actions.shape[0], 1, 1)
       
        tau1 = (Q_plus_unnormed - Q_max_unnormed)/(P_max_unnormed - P_plus_unnormed)
        tau2 = (-Q_plus_unnormed + Q_max_unnormed)/(P_max_unnormed - P_plus_unnormed)

        rho1 = Q_max_unnormed - tau1*P_plus_unnormed
        rho2 = -Q_max_unnormed - tau2*P_plus_unnormed
        h =torch.stack((torch.zeros(actions.shape[0], device = self.device, dtype=torch.float32), P_max_unnormed, P_pots_unnormed, Q_max_unnormed, Q_max_unnormed, rho1, -rho2), 1)
        G[..., -2 , 0] = -tau1
        G[..., -1 , 0] = tau2
        
        sol, lambda_ = self.kinn(torch.stack([actions[...,0], actions[...,1], P_pots, P_max, Q_max, P_plus, Q_plus], 1))

        sol_unnormed = torch.stack([P_pots_unnormed, Q_max_unnormed], 1) * sol

        feasibility = self.constraints(G, h, sol_unnormed)
        #complementarity = torch.where(torch.isclose(torch.zeros_like(feasibility), feasibility), 0.0, lambda_ * feasibility)
        complementarity = lambda_*feasibility
        # grad_L = torch.autograd.grad(self.cost(a, b, r, x_0, x_ref, U), U, grad_outputs=torch.ones_like(U), is_grads_batched=True)[0]
        grad_L = vmap(grad(self.lagrangian, argnums=3), in_dims=(0, 0, 0, 0, 0))(
            actions_unnormed, G, h, sol_unnormed,lambda_
        )

        loss_stationarity = torch.square(grad_L).sum(1)
        loss_feasibility = torch.square(torch.relu(feasibility)).sum(1)
        loss_complementarity = torch.square(complementarity).sum(1)
        return loss_stationarity, loss_feasibility, loss_complementarity

    def training_step(self):
        self.kinn.train()
        sampled_batch = self.sobol_eng.draw(self.batch_size).to(self.device)
        #actions = torch.tensor([1.0, 2.0], device = self.device, dtype=torch.float32) * sampled_batch[..., :2]  + torch.tensor([0.0, -1.0], device = self.device, dtype=torch.float32)
        """ P_max = 0.8*sampled_batch[..., 2] + 0.2
        Q_max =  0.8*sampled_batch[..., 3] + 0.2
        P_plus = (0.9*P_max - 0.1)*sampled_batch[..., 4] + 0.1
        Q_plus = (0.9*Q_max - 0.1)*sampled_batch[..., 5] + 0.1
        P_pots = P_max*sampled_batch[..., 6] """
        actions = 2.*sampled_batch[..., :2]-1.
        P_max = 2.*sampled_batch[..., 2]-1.
        Q_max =  2.*sampled_batch[..., 3]-1.
        P_plus = 2.*sampled_batch[..., 4]-1.
        Q_plus = 2.*sampled_batch[..., 5]-1.
        P_pots = 2.*sampled_batch[..., 6]-1.
        # U, lambda_, mu = self.net(torch.stack([a, b, r, x_0, x_ref, t], 1))

        def closure():
            self.optimizer.zero_grad()
            loss_stationarity, loss_feasibility, loss_complementarity = self.kkt_loss(actions, P_pots, P_max, Q_max, P_plus, Q_plus)
            loss = (loss_stationarity + loss_feasibility+loss_complementarity).mean()
            loss.backward()
            self.tb_logger.add_scalar(
            "Loss/Sum",
            loss,
            self.n_iter,
            )
            self.loss_stationarity.append(loss_stationarity.mean().item())
            self.loss_feasibility.append(loss_feasibility.mean().item())
            self.loss_complementarity.append(loss_complementarity.mean().item())
            self.tb_logger.add_scalar("Loss/Stat", loss_stationarity.mean(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Feas", loss_feasibility.mean(), self.n_iter)
            self.tb_logger.add_scalar("Loss/Comp", loss_complementarity.mean(), self.n_iter)
            self.tb_logger.flush()
            self.n_iter += 1

            if self.es.early_stop(loss):
                if isinstance(self.optimizer, optim.Adam) :
                    print("LBFGS")
                    self.es=EarlyStopper(patience=1000)
                    self.optimizer = optim.LBFGS(self.kinn.parameters(), lr=1e-3)
                else:
                    self.terminated = True
            #self.scheduler.step(loss)
            return loss
        #torchjd.backward([loss_stationarity.mean(), loss_feasibility.mean(), loss_complementarity.mean()], self.kinn.parameters(), A=self.agg)
        
        # torch.nn.utils.clip_grad_norm_(self.solnet.parameters(), 1.0)
        # torch.nn.utils.clip_grad_norm_(self.lambdanet.parameters(), 1.0)
        self.optimizer.step(closure)
        
        
        return self.terminated
    def validation_step(self, X, y):
        self.kinn.eval()
        with torch.no_grad():
            X = X.to(self.device)
            y = y.to(self.device)

            P_max_unnormed = 0.5*0.8*(X[..., 3] +1.0) + 0.2
            Q_max_unnormed =  0.5*0.8*(X[..., 4] +1.0) + 0.2
            P_pots_unnormed = 0.5*(X[..., 2] +1.0) * P_max_unnormed
            sol, lambda_ = self.kinn(X)
            sol_unnormed = torch.stack([P_pots_unnormed, Q_max_unnormed], 1) * sol
            val_loss = nn.functional.l1_loss(sol_unnormed, y)
            val_r2 = R2Score(2).to(self.device)(sol_unnormed, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(sol_unnormed, y)
            rmse = MeanSquaredError(squared=False).to(self.device)(sol_unnormed, y)

            self.r2.append(val_r2.item())
            self.mape.append(val_mape.item())
            self.mae.append(val_loss.item())
            self.rmse.append(rmse.item())
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
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model = KKT_NN()
    pbar = tqdm()
    
    epoch = 0
    for i in range(10):
        pbar.set_description(f"Epoch {epoch+1}")
        terminated = model.training_step()
        for X, y in loader:
            model.validation_step(X, y)
        epoch += 1

        if terminated:
            break
    log = pd.DataFrame({'Step': [i for i in range(1,model.n_iter +1 )], 'Stationarity': model.loss_stationarity, 'Feasibility': model.loss_feasibility, 'Complementarity': model.loss_complementarity}) 
    metrics = pd.DataFrame({'Step': [i for i in range(1,model.n_iter +1 )], 'R2': model.r2, 'MAPE': model.mape, 'MAE': model.mae, 'RMSE': model.rmse}) 
    log.to_csv("Projection/log.csv", index=False)
    metrics.to_csv("Projection/metrics.csv", index=False)
    torch.save(model.kinn.state_dict(), "Projection/kinn.pt")