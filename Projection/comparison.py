import numpy as np
import cvxpy as cp
import torch
import pandas as pd
import random
from cvxpylayers.torch import CvxpyLayer
from pickle import load
from time import time
from tqdm import trange
from KKT_NN import KKT_NN

random.seed(42)     # python random generator
np.random.seed(42)  # numpy random generator
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
device = torch.device("cpu" if torch.cuda.is_available() else "mps")


x = cp.Variable(2)
point = cp.Parameter(2)
h = cp.Parameter(7)
G = cp.Parameter((7, 2))
prob = cp.Problem(cp.Minimize(cp.sum_squares(x - point)), [G @ x <= h])
generator_layer = CvxpyLayer(prob, parameters=[point, G, h], variables=[x])



ds = load(open("Projection/projection.pkl", "rb"))
ds = np.stack([ds[i][0] for i in range(len(ds))])

ds = torch.tensor(ds).to(dtype=torch.float32, device=device)

p, q, p_pot, p_max, q_max, p_plus, q_plus = ds[..., 0], ds[..., 1], ds[..., 2], ds[..., 3], ds[..., 4], ds[..., 5], ds[..., 6]

p = 0.5*(p+1)
p_max = 0.5*0.8*(p_max +1.0) + 0.2
q_max = 0.5*0.8*(q_max +1.0) + 0.2
p_plus = 0.5*(0.9*p_max - 0.1)*(p_plus +1.0) + 0.1
q_plus = 0.5*(0.9*q_max - 0.1)*(q_plus +1.0) + 0.1
p_pot = 0.5*(p_pot+1)*p_max
a = torch.stack([p, q], 1)
tau1 = (q_plus - q_max)/(p_max - p_plus)
tau2 = (-q_plus + q_max)/(p_max - p_plus)
rho1 = q_max - tau1*p_plus
rho2 = -q_max - tau2*p_plus

G_val = torch.Tensor(
                [
                    [-1, 0], [1, 0], [1, 0], [0, -1], [0, 1], [0, 1], [0, -1]
                ]
            ).to(dtype=torch.float32, device=device).repeat(len(a), 1, 1)

G_val[..., -2 , 0] = tau1
G_val[..., -1 , 0] = tau2
h_val =torch.stack((torch.zeros(len(a), device = device, dtype=torch.float32), p_max, p_pot, q_max, q_max, rho1, -rho2), 1)

model = KKT_NN()
model.kinn.load_state_dict(torch.load("Projection/kinn.pt", map_location = device))
model.kinn.to(device)

kinn_times = []
cvxpy_times = []
for i in trange(1, 20):
    start_kinn = time()
    pred = model.kinn(ds[:i, ...])[0]
    end_kinn = time()

    start_cvxpy= time()
    gt = generator_layer(a[:i, ...], G_val[:i, ...], h_val[:i, ...])[0]
    end_cvxpy = time()
    kinn_times.append(end_kinn-start_kinn)
    cvxpy_times.append(end_cvxpy-start_cvxpy)
report = pd.DataFrame({"Batch size": np.array([i for i in range(1, 20)]), "KINN": np.array(kinn_times), "CVXPY": np.array(cvxpy_times)})

report.to_csv("Projection/times.csv", index = False)