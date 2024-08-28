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
    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = (
            nn.Sequential(
                nn.Linear(6, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 18),
            ).to(dtype=torch.float32, device=device)
        )

    def forward(self, X):
        y = self.mlp(X)

        sol = torch.nn.functional.softmax(y[..., :-2], dim = 1)
        lambd = torch.nn.functional.relu(y[..., 4:])

        return y[..., :4], lambd


class KKT_NN(L.LightningModule):

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = Net()
        self.G_val = torch.Tensor(
                [
                    [-1, 0, 0, 0],
                    [1, 0, 0 ,0],
                    [1, 0, 0, 0],
                    [0, 0, -1 ,0],
                    [0, 0, 1 ,0],
                    [1.5, 0, 1,0],
                    [1.5, 0, -1, 0],
                    [0, -1,0, 0],
                    [0, 1, 0 ,0],
                    [0, 1, 0, 0],
                    [0, 0, 0 ,-1],
                    [0, 0, 0 ,1],
                    [0, 2,0 ,1],
                    [0, 2, 0, -1],
                ]
            ).to(dtype=torch.float32, device=device)
        
        self.h_val = torch.tensor(
                [
                    [-0.0000, 0.3000, 0.0400, 0.3000, 0.3000, 0.6000, 0.6000, -0.0000, 0.5000,
        0.1100, 0.5000, 0.5000, 1.2000, 1.2000]
                ],
            ).to(dtype=torch.float32, device=device)
        self.alpha = 0.999
        self.beta_p = 1.0
        self.tau = 1e-3

        self.coeffs = torch.ones(4, device=device)
        self.initial_losses = None
        self.previous_losses = None
    
    def coeff_rel_improv(self,losses, prev_losses):
         with torch.no_grad():
            coeff_rel_improv = 5*torch.exp(losses/(self.tau*prev_losses + torch.finfo(torch.float32).eps) - torch.max(losses/(self.tau*prev_losses+ torch.finfo(torch.float32).eps)))

            coeff_rel_improv /= torch.sum(torch.exp(losses/(self.tau*prev_losses + torch.finfo(torch.float32).eps) - torch.max(losses/(self.tau*prev_losses+ torch.finfo(torch.float32).eps))))

            return coeff_rel_improv
    def kkt_loss(self, actions, P_pots, sol, lambd,):

        beta = torch.bernoulli(torch.tensor(self.beta_p))

        h_val =self.h_val.repeat(actions.shape[0], 1)

        h_val[..., 2] = P_pots[..., 0]
        h_val[..., 9] = P_pots[..., 1]
        grad_g = torch.matmul(self.G_val.T, lambd.T).T
        grad_f = sol - actions
        g_ineq = torch.matmul(self.G_val, sol.T).T - h_val
        dual_feasibility = torch.relu(-lambd)
        stationarity = grad_f + grad_g
        complementary = lambd * g_ineq
        loss_stationarity = torch.norm(stationarity)
        loss_g_ineq = torch.norm(torch.relu(g_ineq))
        loss_complementary = torch.norm(complementary)
        loss_dual_feasibility = 0*torch.norm(dual_feasibility)
        loss_sparsity = torch.norm(sol, p=1)

    
        losses = torch.stack([loss_stationarity, loss_g_ineq, loss_complementary, loss_dual_feasibility])
        
        if self.initial_losses is None:
            self.initial_losses = losses

        if self.previous_losses is None:
            self.previous_losses = losses

        self.coeffs = self.alpha*(beta*self.coeffs + (1-beta)*self.coeff_rel_improv(losses, self.initial_losses)) + (1-self.alpha)*self.coeff_rel_improv(losses, self.previous_losses)
        
        self.previous_losses = losses

        return (
            self.coeffs@losses + 0.0*loss_sparsity,
            loss_stationarity,
            loss_g_ineq,
            loss_complementary,
            loss_dual_feasibility,
        )

    def training_step(self, batch, batch_idx):
        X,y = batch
        #X = torch.rand(64).to(self.device) * 0.3
        #X = X.unsqueeze(1)
        actions = X[..., :4]
        P_pots = X[...,4:]
        sol, lambd = self.net(X)
        kkt_loss, stationarity, g_ineq, complementary, feasability = (
            self.kkt_loss(actions, P_pots, sol, lambd)
        )
        self.log_dict(
            {
                "train_loss": kkt_loss,
                "stationarity": stationarity,
                "g_ineq": g_ineq,
                "complementary": complementary,
                "feasability": feasability,
            },
            prog_bar=True,
        )
        return kkt_loss

    def validation_step(self, batch, batch_idx):

        X, y = batch
        sol, lambd = self.net(X)
        val_loss = nn.functional.mse_loss(sol, y)
        val_r2 = R2Score(4).to(self.device)(sol, y)
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_r2", val_r2, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.mlp(X)
        test_mse = nn.functional.mse_loss(y_hat, y)
        test_r2 = R2Score().to(self.device)(y_hat.view(-1), y.view(-1))
        self.log("test_mse", test_mse)
        self.log("test_r2", test_r2)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=3e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, threshold=1e-6
        )
        return {"optimizer": optimizer}


class Samples(Dataset):
    def __init__(self, transform=None):
        self.samples = load(open("KKT-NN/dataset_loads.pkl", "rb"))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]
        X = np.array(X)[..., [0,1,2,3,6,7]]
        y = np.array(y)[..., [0,1,3,4]]
        if self.transform:
            X = self.transform(X)
            y = self.transform(y)
        return X.astype(np.float32), y.astype(np.float32)

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
    logger = TensorBoardLogger("KKT-NN/tb_logs", name="my_model")
    trainer = L.Trainer(max_epochs=10000, callbacks=[], precision="32", logger=logger)
    trainer.fit(model, train_loader, val_loader)
    # res = trainer.test(model, test_loader)

    dump(res, open("res.pt", "wb"))
