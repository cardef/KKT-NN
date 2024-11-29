import numpy as np
import cvxpy as cp
import scipy.linalg
import torch
import pandas as pd
import scipy
from cvxpylayers.torch import CvxpyLayer
from pickle import load
from time import time
from tqdm import trange
from KKT_NN import KKT_NN


device = torch.device("cpu" if torch.cuda.is_available() else "mps")

a_min = 0.7
a_var = 0.3

b_min = 0.1
b_var = 0.4

r_min = 0.0
r_var = 1.0

x_0_min = 0.0
x_0_var = 1.0

x_ref_min = 0.0
x_ref_var = 1.0

horizon = 10



ds = load(open("MPC/mpc.pkl", "rb"))
ds = np.stack([ds[i][0] for i in range(len(ds))])

ds = torch.tensor(ds).to(dtype=torch.float32, device=device)

a, b, r, x_0, x_ref= ds[..., 0], ds[..., 1], ds[..., 2], ds[..., 3], ds[..., 4]

a=0.5*a_var*(a + 1)+ a_min
b=0.5*b_var*(b  + 1)+ b_min
r = 0.5*r_var*(r + 1) + r_min
x_0_val=  0.5*x_0_var*(x_0 + 1) + x_0_min
x_ref_val = 0.5*x_ref_var*(x_ref + 1) + x_ref_min
L_val = torch.eye(horizon).unsqueeze(0).repeat(r.shape[0], 1, 1)
for i in range(r.shape[0]):
    L_val[i,...] *= r[i]

L_sqrt_val = torch.zeros_like(L_val)
for i in range(r.shape[0]):
    L_sqrt_val[i,...] = torch.tensor(scipy.linalg.sqrtm(L_val[i, ...]))
I = torch.eye(horizon)
A_val = torch.tensor([[a[j]**(i+1) for i in range(horizon)] for j in range(a.shape[0])])
B_val = torch.tensor([[[a[k]**(i-1) * b[k] for i in range(j, 0, -1)] + [0]*(horizon-j) for j in range(1, horizon+1)] for k in range(a.shape[0])])

u = cp.Variable(horizon, nonneg = True)
x_next = cp.Variable(horizon, nonneg = True)
x_next_prov = cp.Parameter(horizon, nonneg = True)
L_sqrt = cp.Parameter((horizon, horizon), PSD=True)
A = cp.Parameter(horizon, nonneg = True)
B = cp.Parameter((horizon, horizon), PSD = True)
x_0 = cp.Parameter( nonneg = True)
x_ref = cp.Parameter( nonneg = True)
constraints = [u <= np.ones(horizon), -u <= np.zeros(horizon), (x_next) <= np.ones(horizon), -(x_next) <= np.zeros(horizon), x_next == B@u + x_next_prov]
prob = cp.Problem(cp.Minimize(cp.quad_form(x_next - x_ref, I) + cp.sum_squares(L_sqrt@u)), constraints)

mpc_layer = CvxpyLayer(prob, parameters=[B, L_sqrt, x_next_prov, x_ref], variables=[u, x_next]).to(device)
model = KKT_NN()
#model.kinn.load_state_dict(torch.load("MPC/mpc.pt", map_location = device))
model.kinn.to(device)

kinn_times = []
cvxpy_times = []

max_batch_size = 100
for i in trange(1, max_batch_size):
    start_kinn = time()
    pred = model.kinn(ds[:i, ...])[0]
    end_kinn = time()
    
    x_next_prov_val = A_val[:i,...]*x_0_val.unsqueeze(1)[:i,...]
    start_cvxpy= time()
    gt = mpc_layer( B_val[:i,...], L_sqrt_val[:i ,...], x_next_prov_val, x_ref_val[:i,...])[0]
    end_cvxpy = time()
    kinn_times.append(end_kinn-start_kinn)
    cvxpy_times.append(end_cvxpy-start_cvxpy)
report = pd.DataFrame({"Batch size": np.array([i for i in range(1, max_batch_size)]), "KINN": np.array(kinn_times), "CVXPY": np.array(cvxpy_times)})

report.to_csv("MPC/times.csv", index = False)