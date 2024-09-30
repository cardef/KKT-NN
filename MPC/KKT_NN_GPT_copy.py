import torch
import numpy as np
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score, MeanAbsolutePercentageError
import torchjd
from torchjd.aggregation import UPGrad, NashMTL, DualProj, GradDrop
from datetime import datetime
from tqdm import tqdm
from pickle import load
import time


class EarlyStopper:
    def __init__(self, patience=100, min_delta=1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss - self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class ResidualBlock(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(n, n)
        self.relu = nn.LeakyReLU()
        self.ln = nn.BatchNorm1d(n)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X):
        identity = X
        out = self.linear(X)
        out = self.ln(out)
        out = self.relu(out + identity)
        out = self.dropout(out)
        return out

class Net(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.softplus = nn.Softplus(beta=1, threshold=20)  # Parametri regolabili
        self.mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, self.n + 2),
        ).to(dtype=torch.float32)

        # Inizializza i pesi dei moltiplicatori con piccoli valori positivi
        """ with torch.no_grad():
            self.mlp[-1].weight[:, 1000:] *= 0.01  # Riduci l'impatto iniziale
            self.mlp[-1].bias[:, 1000:] = 0.01 """

    def forward(self, X):
        y = self.mlp(X)
        x = y[..., : self.n]
        lambda_ = self.softplus(y[..., self.n : self.n + 1]).squeeze()
        mu = self.softplus(y[..., self.n + 1 :]).squeeze()
        return x, lambda_, mu


class KKT_NN:
    def __init__(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.horizon = 1000

        
        self.net = Net(self.horizon).to(self.device)

        self.batch_size = 64
        self.n_iter = 0
        self.tolerance = 1e-6
        self.soboleng = torch.quasirandom.SobolEngine(dimension=2)
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1000, verbose=True
        )

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        self.tb_logger = SummaryWriter("Smoothing/runs/tb_logs/" + current_time)

        self.D = np.eye(self.n) - np.eye(self.n, k=-1)
        self.D = self.D[:-1, :]  # Prima derivata discreta
        self.D = torch.tensor(self.D, dtype=torch.float32, device=self.device)

        self.D2 = np.eye(self.n) - 2 * np.eye(self.n, k=1) + np.eye(self.n, k=2)
        self.D2 = self.D2[:-2, :]
        self.D2 = torch.tensor(self.D2, dtype=torch.float32, device=self.device)
        self.es = EarlyStopper(patience=100, min_delta=1e-2)
        self.plateau = False
        self.terminated = False

        # Weights for loss components
        self.weight_stationarity = 1.0
        self.weight_concavity_feasibility = 1.0
        self.weight_variation_feasibility = 1.0
        self.complementarity_weight_concavity = (
            1.0  # Incrementato per maggiore importanza
        )
        self.complementarity_weight_variation = (
            1.0  # Incrementato per maggiore importanza
        )

        # Regolarizzazione L1 per favorire la sparsità (inizialmente disabilitata)
        self.lambda_reg_weight = 1e-5
        self.mu_reg_weight = 1e-5

    def objective_fun(self, x, signal):
        return torch.square(x - signal).sum(1)

    def concavity_constraint(self, x, kappa, signal):
        return torch.square(torch.matmul(self.D2, x.T)).T.sum(1) - (
            kappa * torch.square(torch.matmul(self.D2, signal.T)).T.sum(1)
        )

    def variation_constraint(self, x, rho, signal):
        return torch.square(torch.matmul(self.D, x.T)).T.sum(1) - (
            rho * torch.square(torch.matmul(self.D, signal.T)).T.sum(1)
        )

    def kkt_loss(self, x, rho, kappa, lambda_, mu, signal):
        """
        Compute the KKT loss components:
        1. Stationarity
        2. Concavity Feasibility
        3. Variation Feasibility
        4. Complementarity for Concavity
        5. Complementarity for Variation
        6. L1 Regularization for Sparsity (optional)
        """
        # Objective function
        cost = self.objective_fun(x, signal)  # Shape: (batch_size,)

        # Constraints
        concavity = self.concavity_constraint(x, kappa, signal)  # Shape: (batch_size,)
        variation = self.variation_constraint(x, rho, signal)  # Shape: (batch_size,)

        # Compute gradients
        cost_mean = cost.mean()
        concavity_mean = concavity.mean()
        variation_mean = variation.mean()

        grad_f = torch.autograd.grad(cost_mean, x, create_graph=True)[
            0
        ]  # Shape: (batch_size, n)
        grad_g2 = torch.autograd.grad(concavity_mean, x, create_graph=True)[
            0
        ]  # Shape: (batch_size, n)
        grad_g1 = torch.autograd.grad(variation_mean, x, create_graph=True)[
            0
        ]  # Shape: (batch_size, n)

        # Identify active constraints using tolerance
        active_concavity = concavity.abs() <= self.tolerance  # Shape: (batch_size,)
        active_variation = variation.abs() <= self.tolerance  # Shape: (batch_size,)

        # Set Lagrange multipliers to zero if constraints are inactive
        lambda_active = torch.where(
            active_variation, lambda_, torch.zeros_like(lambda_)
        )  # Shape: (batch_size,)
        mu_active = torch.where(
            active_concavity, mu, torch.zeros_like(mu)
        )  # Shape: (batch_size,)

        # Expand multipliers for broadcasting
        lambda_active = lambda_active.unsqueeze(1)  # Shape: (batch_size, 1)
        mu_active = mu_active.unsqueeze(1)  # Shape: (batch_size, 1)

        # Compute the gradient of the Lagrangian
        grad_L = (
            grad_f + lambda_active * grad_g1 + mu_active * grad_g2
        )  # Shape: (batch_size, n)

        # Stationarity condition: ||grad_L||^2
        stationarity = torch.square(grad_L).sum(1).mean()  # Scalar

        # Feasibility: Penalize constraint violations
        concavity_feasibility = torch.relu(concavity).mean()  # Scalar
        variation_feasibility = torch.relu(variation).mean()  # Scalar

        # Complementarity: Penalize lambda >0 and g_i(x) <0
        complementarity_concavity = (
            (mu_active.squeeze(1) * torch.relu(-concavity)).pow(2).mean()
        )  # Scalar
        complementarity_variation = (
            (lambda_active.squeeze(1) * torch.relu(-variation)).pow(2).mean()
        )  # Scalar

        # Regolarizzazione L1 per favorire la sparsità dei moltiplicatori
        lambda_reg = torch.abs(lambda_).mean()
        mu_reg = torch.abs(mu).mean()

        # Aggiungi la regolarizzazione alla loss
        # complementarity_concavity += self.lambda_reg_weight * lambda_reg
        # complementarity_variation += self.mu_reg_weight * mu_reg

        return (
            stationarity,
            concavity_feasibility,
            variation_feasibility,
            complementarity_concavity,
            complementarity_variation,
        )

    def training_step(self):
        # Campionamento di rho e kappa che aumentano la probabilità di vincoli attivi
        self.net.train()
        signal = self.true + 0.1 * torch.rand(
            (self.batch_size, self.n), dtype=torch.float32, device=self.device
        ) 
        rho_kappa = self.soboleng.draw(self.batch_size, dtype=torch.float32).to(self.device)
        rho = rho_kappa[..., 0]  # Valori tra 0.75 e 1.25
        kappa = rho_kappa[..., 1]  # Valori tra 0.75 e 1.25

        # Forward pass
        x, lambda_, mu = self.net(signal, rho, kappa)

        # Calcolo della loss KKT
        (
            stationarity,
            concavity_feasibility,
            variation_feasibility,
            concavity_complementary,
            variation_complementary,
        ) = self.kkt_loss(x, rho, kappa, lambda_, mu, signal)

        # Calcolo della loss totale con pesi
        total_loss = (
            self.weight_stationarity * stationarity
            + self.weight_concavity_feasibility * concavity_feasibility
            + self.weight_variation_feasibility * variation_feasibility
            + self.complementarity_weight_concavity * concavity_complementary
            + self.complementarity_weight_variation * variation_complementary
        )
        losses = [self.weight_stationarity * stationarity
            ,self.weight_concavity_feasibility * concavity_feasibility
            ,self.weight_variation_feasibility * variation_feasibility
            ,self.complementarity_weight_concavity * concavity_complementary
            ,self.complementarity_weight_variation * variation_complementary]
        # Backward pass e ottimizzazione
        self.optimizer.zero_grad()
        total_loss.backward()
        #torchjd.backward(losses, self.net.parameters(), self.agg)
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1e-2)
        self.optimizer.step()
        self.scheduler.step(total_loss)

        # Early Stopping
        if self.es.early_stop(total_loss.item()):
            self.plateau = True
        if self.plateau:
            if isinstance(self.optimizer, optim.LBFGS):
                self.terminated = True

        # Logging dei componenti della loss
        self.tb_logger.add_scalar(
            "Loss/Sum",
            total_loss.item(),
            self.n_iter,
        )
        self.tb_logger.add_scalar("Loss/Stat", stationarity.item(), self.n_iter)
        self.tb_logger.add_scalar(
            "Loss/Cost", self.objective_fun(x, signal).mean().item(), self.n_iter
        )
        self.tb_logger.add_scalar(
            "Loss/Conc", concavity_feasibility.item(), self.n_iter
        )
        self.tb_logger.add_scalar("Loss/Var", variation_feasibility.item(), self.n_iter)
        self.tb_logger.add_scalar(
            "Loss/ConcComp", concavity_complementary.item(), self.n_iter
        )
        self.tb_logger.add_scalar(
            "Loss/VarComp", variation_complementary.item(), self.n_iter
        )

        # Logging dei moltiplicatori di Lagrange
        self.tb_logger.add_histogram(
            "Lambda", lambda_.detach().cpu().numpy(), self.n_iter
        )
        self.tb_logger.add_histogram("Mu", mu.detach().cpu().numpy(), self.n_iter)

        # Logging delle violazioni dei vincoli
        concavity_violation = (concavity_feasibility > 0).float().mean().item()
        variation_violation = (variation_feasibility > 0).float().mean().item()
        self.tb_logger.add_scalar(
            "Concavity_Violation_Percentage", concavity_violation, self.n_iter
        )
        self.tb_logger.add_scalar(
            "Variation_Violation_Percentage", variation_violation, self.n_iter
        )

        self.tb_logger.flush()
        self.n_iter += 1
        return total_loss.item()

    def validation_step(self, signal, rho, kappa, y, mu, lambda_):
        self.net.eval()
        y = y.to(self.device)
        mu = mu.to(self.device).squeeze()
        lambda_ = lambda_.to(self.device).squeeze()
        rho = rho.to(self.device)
        kappa = kappa.to(self.device)
        signal = signal.to(self.device)
        with torch.no_grad():
            x, pred_lambda, pred_mu = self.net(signal, rho, kappa)
            val_loss = nn.functional.l1_loss(x, y)
            val_r2 = R2Score(self.n).to(self.device)(x, y)
            val_mape = MeanAbsolutePercentageError().to(self.device)(x, y)

        # Calcolo delle condizioni KKT

        y = y.to(self.device).requires_grad_()
        
        (
            stationarity,
            concavity_feasibility,
            variation_feasibility,
            concavity_complementary,
            variation_complementary,
        ) = self.kkt_loss(y, rho, kappa, lambda_, mu, signal)

        # Logging della validazione
        self.tb_logger.add_scalar("Val/Loss", val_loss.item(), self.n_iter)
        self.tb_logger.add_scalar("Val/R2", val_r2.item(), self.n_iter)
        self.tb_logger.add_scalar("Val/Stat", stationarity.item(), self.n_iter)
        self.tb_logger.add_scalar("Val/Conc", concavity_feasibility.item(), self.n_iter)
        self.tb_logger.add_scalar("Val/Var", variation_feasibility.item(), self.n_iter)
        self.tb_logger.add_scalar(
            "Val/ConcComp", concavity_complementary.item(), self.n_iter
        )
        self.tb_logger.add_scalar(
            "Val/VarComp", variation_complementary.item(), self.n_iter
        )
        self.tb_logger.add_scalar(
            "Val/Lambda_Max", pred_lambda.max().item(), self.n_iter
        )
        self.tb_logger.add_scalar("Val/Mu_Max", pred_mu.max().item(), self.n_iter)
        self.tb_logger.flush()


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("Smoothing/smooth.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        signal, rho, kappa, y, mu, lambda_ = self.samples[idx]
        if self.transform:
            y = self.transform(y)
        return (
            signal.astype(np.float32),
            rho.astype(np.float32),
            kappa.astype(np.float32),
            y.astype(np.float32),
            mu.astype(np.float32),
            lambda_.astype(np.float32),
        )


if __name__ == "__main__":
    dataset = Samples()
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model = KKT_NN()
    pbar = tqdm(total=10000, desc="Training")

    for epoch in range(10000):
        pbar.set_description(f"Epoch {epoch+1}")
        model.training_step()
        if epoch % 100 == 0:
            for signal, rho, kappa, y, mu, lambda_ in loader:
                model.validation_step(signal, rho, kappa, y, mu, lambda_)
        pbar.update(1)
