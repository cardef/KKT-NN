import lightning as L
import pandas as pd
import numpy as np
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
from torch.utils.data import Dataset, DataLoader, random_split
from pickle import load, dump


class Net(nn.Module):
    def softmax(self, input, t=1e-2):
        ex = torch.exp(input/t - torch.max(input/t, 1)[0].unsqueeze(1))
        sum = torch.sum(ex, axis=1).unsqueeze(1)
        return ex / sum
    def __init__(self, channels):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear1 = nn.Linear(channels, 512)
        self.linear2 = nn.Linear(512, 512)
        self.linear3 = nn.Linear(512, 512)
        self.linear4 = nn.Linear(512, channels+1)
        self.relu = nn.ReLU()
        self.mlp = (
            nn.Sequential(
                nn.Linear(channels, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, channels+1),
            ).to(dtype=torch.float64, device=device)
        )

    def forward(self, X):
        # y = self.relu(self.linear1(X))
        # y = self.relu(self.linear2(y))
        # y = self.relu(self.linear3(y))
        # y = self.linear4(y)
        y=self.mlp(X)
        sol = torch.nn.functional.relu(y[..., :-1])
        nu = y[..., [-1]]

        return sol, nu


class KKT_NN(L.LightningModule):

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.channels=20
        self.net = Net(self.channels)
        self.alpha = 0.999
        self.beta_p = 0.9999
        self.tau = 1e-5

        self.coeffs = torch.ones(2, device=device, dtype = torch.float64)
        self.initial_losses = None
        self.previous_losses = None
    def loss_grad_std_wn(self, loss, net):
        with torch.no_grad():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            grad_ = torch.zeros((0), dtype=torch.float64, device=device)
            for elem in torch.autograd.grad(loss, net.parameters(), retain_graph=True, allow_unused=False):
                grad_ = torch.cat((grad_, elem.view(-1)))
            
            return 1/torch.std(grad_)
    def coeff_rel_improv(self,losses, prev_losses):
         with torch.no_grad():
            coeff_rel_improv = 2*torch.exp(losses/(self.tau*prev_losses + torch.finfo(torch.float64).eps) - torch.max(losses/(self.tau*prev_losses+ torch.finfo(torch.float64).eps)))

            coeff_rel_improv /= torch.sum(torch.exp(losses/(self.tau*prev_losses + torch.finfo(torch.float64).eps) - torch.max(losses/(self.tau*prev_losses+ torch.finfo(torch.float64).eps))))

            return coeff_rel_improv
    def kkt_loss(self, x, sol, nu):

        beta = torch.bernoulli(torch.tensor(self.beta_p))

        grad_f = -1.0/(x + sol)
        g_eq = sol.sum(dim=1) - 1.0
        stationarity = grad_f + nu * torch.ones_like(sol)
        lagrangian = -torch.log(x+sol)
        loss_stationarity = torch.mean(torch.abs(stationarity))
        loss_g_eq = torch.mean(torch.abs(g_eq))
        loss_sparsity = torch.norm(sol, p=1)

    
        losses = torch.stack([loss_stationarity, loss_g_eq])
        
        if self.initial_losses is None:
            self.initial_losses = losses

        if self.previous_losses is None:
            self.previous_losses = losses

        self.coeffs = self.alpha*(beta*self.coeffs + (1-beta)*self.coeff_rel_improv(losses, self.initial_losses)) + (1-self.alpha)*self.coeff_rel_improv(losses, self.previous_losses)
        self.coeffs = self.coeffs/self.coeffs[0]
        #for i in range(1, losses.shape[0]):
        #    self.coeffs[i] = self.alpha*self.coeffs[i]+(1-self.alpha)*(self.loss_grad_std_wn(losses[i], self.net)/(self.loss_grad_std_wn(losses[0], self.net) + torch.finfo(torch.float64).eps))
        self.previous_losses = losses
        #self.coeffs[1] = 0.0
        print(self.coeffs)
        return (
            self.coeffs@losses,
            torch.mean(loss_stationarity),
            loss_g_eq,
        )

    def training_step(self, batch, batch_idx):

        X = (1.5*torch.rand(512, self.channels) + 0.5).to(self.device)
        sol, nu = self.net(X)
        kkt_loss, stationarity, g_eq = (
            self.kkt_loss(X, sol, nu)
        )
        self.log_dict(
            {
                "train_loss": kkt_loss,
                "stationarity": stationarity,
                "g_eq": g_eq,
                "coeff_stationarity": self.coeffs[0],
                "coeff_g_eq": self.coeffs[1],
            },
            prog_bar=True,
        )
        return kkt_loss

    def validation_step(self, batch, batch_idx):

        X, y = batch
        print(X.shape, y.shape)
        sol, nu = self.net(X)
        val_loss = nn.functional.mse_loss(sol, y)
        val_r2 = R2Score(self.channels).to(self.device)(sol, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_r2", val_r2, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.mlp(X)
        test_mse = nn.functional.mse_loss(y_hat, y)
        print(y_hat.view(-1))
        test_r2 = R2Score().to(self.device)(y_hat.view(-1), y.view(-1))
        self.log("test_mse", test_mse)
        self.log("test_r2", test_r2)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.net.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, threshold=1e-6
        )
        return {"optimizer": optimizer}


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("KKT-NN/water_filling/water_filling.pkl", "rb"))
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

if __name__ == '__main__':
    dataset = Samples()
    train_set, val_set = random_split(dataset, [0.1, 0.9])
    train_loader = DataLoader(train_set, 512, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset, 512, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_set, 512, shuffle=False, num_workers=5)
    model = KKT_NN()

    early_stop_callback = EarlyStopping(
        monitor="train_loss", min_delta=0.00, patience=10, verbose=False, mode="min"
    )
    logger = TensorBoardLogger("KKT-NN/water_filling/tb_logs", name="my_model")
    trainer = L.Trainer(max_epochs=10000, callbacks=[], precision="64", logger=logger, accelerator="cpu")
    trainer.fit(model, train_loader, val_loader)
    # res = trainer.test(model, test_loader)

    dump(res, open("res.pt", "wb"))
