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


class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Se il numero di canali di input e output differisce, aggiungi una convoluzione di skip
        if in_channels != out_channels:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        if self.skip_conv:
            identity = self.skip_conv(identity)

        out += identity
        out = self.relu(out)
        return out


class ConditionalAutoencoder(nn.Module):
    def __init__(self, signal_length, num_residual_blocks=4, hidden_channels=64, device = torch.device("cpu")):
        super(ConditionalAutoencoder, self).__init__()
        self.signal_length = signal_length
        
        # Encoder
        self.encoder_fc = nn.Sequential(
            nn.Linear(2, 128),  # \rho e \kappa
            nn.LeakyReLU(),
            nn.Linear(128, hidden_channels)
        )
        
        self.encoder_conv = nn.Sequential(
            ConvResidualBlock(1, hidden_channels, kernel_size=3, padding=1),
            *[ConvResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)]
        )
        
        # Decoder
        self.decoder_conv = nn.Sequential(
            ConvResidualBlock(2* hidden_channels, hidden_channels, kernel_size=3, padding=1),
            *[ConvResidualBlock(hidden_channels, hidden_channels) for _ in range(num_residual_blocks)],
            ConvResidualBlock(hidden_channels, hidden_channels + 1, kernel_size=3, padding=1)
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_channels, 128),
            nn.LeakyReLU(),
            nn.Linear(128, hidden_channels),
            nn.LeakyReLU(),
            nn.Linear(hidden_channels, signal_length)
        )
        
        # Moltiplicatori di Lagrange
        self.fc_lambda = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Garantisce valori positivi
        )
        
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Garantisce valori positivi
        )
        
    def forward(self, noisy_signal, rho, kappa):
        """
        X: [batch_size, 2] - Parametri \rho e \kappa
        noisy_signal: [batch_size, N] - Segnale rumoroso
        """
        batch_size = rho.size(0)
        
        # Codifica dei parametri condizionali
        encoded_params = self.encoder_fc(torch.stack([rho, kappa], 1))  # [batch_size, hidden_channels]
        encoded_params = encoded_params.unsqueeze(-1).repeat(1, 1, self.signal_length)  # [batch_size, hidden_channels, 1]
        
        # Preparazione del segnale rumoroso per la convoluzione
        noisy_signal = noisy_signal.unsqueeze(0)  # [batch_size, 1, N]
        
        # Concatenazione delle features condizionali con il segnale rumoroso
        
        # Encoder convoluzionale
        encoded = self.encoder_conv(noisy_signal)  # [batch_size, hidden_channels, N]
        
        # Decoder convoluzionale
        decoded = self.decoder_conv(torch.cat([encoded.unsqueeze(0).repeat(batch_size, 1, 1), encoded_params], 1))  # [batch_size, hidden_channels +1, N]
        
        # Decoder completamente connesso
        decoded = decoded[:, :-1, :].mean(dim=2)  # Aggregazione lungo la dimensione spaziale
        reconstructed_signal = self.decoder_fc(decoded)  # [batch_size, N]
        
        # Moltiplicatori di Lagrange
        lambda_ = self.fc_lambda(decoded)  # [batch_size, 1]
        mu = self.fc_mu(decoded)            # [batch_size, 1]
        
        return reconstructed_signal, lambda_.squeeze(), mu.squeeze()


class Net(nn.Module):
    def __init__(self, signal, n):
        super().__init__()
        self.signal = signal
        self.n = n
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
        self.n = 100
        t = torch.linspace(0, 1, self.n, dtype=torch.float32)
        self.true = (0.8 * ((torch.sin(50 * t) + torch.sin(10 * t)) / 2) + 0.1 * t).to(
            dtype=torch.float32, device=self.device
        )  # Profilo di elevazione reale
        self.noisy = self.true + 0.1 * torch.rand(
            self.n, dtype=torch.float32, device=self.device
        )  # Dati misurati con rumore

        
        self.net = ConditionalAutoencoder(self.n, num_residual_blocks=2, hidden_channels=128, device=self.device).to(self.device)

        self.batch_size = 64
        self.n_iter = 0
        self.tolerance = 1e-6
        self.soboleng = torch.quasirandom.SobolEngine(dimension=2)
        self.agg = UPGrad()
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-5)
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

    def objective_fun(self, x):
        return torch.square(x - self.noisy).sum(1)

    def concavity_constraint(self, x, kappa):
        return torch.square(torch.matmul(self.D2, x.T)).T.sum(1) - (
            kappa * torch.square(torch.matmul(self.D2, self.noisy)).sum()
        )

    def variation_constraint(self, x, rho):
        return torch.square(torch.matmul(self.D, x.T)).T.sum(1) - (
            rho * torch.square(torch.matmul(self.D, self.noisy)).sum()
        )

    def kkt_loss(self, x, rho, kappa, lambda_, mu):
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
        cost = self.objective_fun(x)  # Shape: (batch_size,)

        # Constraints
        concavity = self.concavity_constraint(x, kappa)  # Shape: (batch_size,)
        variation = self.variation_constraint(x, rho)  # Shape: (batch_size,)

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
        rho_kappa = self.soboleng.draw(self.batch_size, dtype=torch.float32).to(self.device)
        rho = rho_kappa[..., 0]  # Valori tra 0.75 e 1.25
        kappa = rho_kappa[..., 1]  # Valori tra 0.75 e 1.25

        # Forward pass
        x, lambda_, mu = self.net(self.noisy, rho, kappa)

        # Calcolo della loss KKT
        (
            stationarity,
            concavity_feasibility,
            variation_feasibility,
            concavity_complementary,
            variation_complementary,
        ) = self.kkt_loss(x, rho, kappa, lambda_, mu)

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
            "Loss/Cost", self.objective_fun(x).mean().item(), self.n_iter
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
        ) = self.kkt_loss(y, rho, kappa, lambda_, mu)

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
        rho, kappa, y, mu, lambda_ = self.samples[idx]
        if self.transform:
            y = self.transform(y)
        return (
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
        for rho, kappa, y, mu, lambda_ in loader:
            model.validation_step(model.noisy, rho, kappa, y, mu, lambda_)
        pbar.update(1)
