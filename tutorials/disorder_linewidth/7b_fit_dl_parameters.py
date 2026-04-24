import sys

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ase.io import read
from scipy.interpolate import interp1d

from tqdm import tqdm

from smooth_disorder.structural import obtain_density, THzToCm, THz, Angstrom
from smooth_disorder.vis.interactive import *

from smooth_disorder.disorder_linewidth import lorentzian_numpy, lorentzian_torch
from smooth_disorder.disorder_linewidth import prepare_fitting_inputs
from smooth_disorder.disorder_linewidth import evaluate_linewidth_and_model_prediction
from smooth_disorder.disorder_linewidth import PDCModel

import torch


CRYSTAL_POSCAR    = "./1_graphite/POSCAR"
DISORDERED_POSCAR = "./2_irg_t2/irg_t2_14009.vasp"

WORK_DIR = "./dl_workflow"

CRYSTAL_VEL_SAVE     = f"{WORK_DIR}/crystal_vdos_group_vel"
DISORDERED_VDOS_SAVE = f"{WORK_DIR}/disordered_vdos"
SHIFTED_SAVE         = f"{WORK_DIR}/reduced_density_crystal_vdos_group_vel"

MODEL_PARAMETERS_SAVE = f"{WORK_DIR}/model_parameters"


# Initial parameter values [nm], [1e-6 THz cm nm^3]
L0, R0 = 3.3, 5.54



# read the input data + setup torch arrays

(density_crystal,
density_disordered,
freq_disordered,
vdos_disordered,
interp_shifted_freq_crystal,
interp_shifted_vdos_crystal,
interp_shifted_speed_crystal) = prepare_fitting_inputs(
    CRYSTAL_POSCAR,
    DISORDERED_POSCAR,
    DISORDERED_VDOS_SAVE,
    SHIFTED_SAVE,
)

# X and Y are the normalised crystal and disordered VDOS used in the loss
X = torch.from_numpy(interp_shifted_vdos_crystal / density_crystal)
Y = torch.from_numpy(vdos_disordered / density_disordered)


# instantiate the model for fitting the linewidths and setup LBFGS search
model = PDCModel(
    L0, R0,
    density_crystal, density_disordered,
    freq_disordered,
    interp_shifted_freq_crystal,
    interp_shifted_vdos_crystal,
    interp_shifted_speed_crystal,
)

optim = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=50,
    line_search_fn="strong_wolfe",
)

loss_fn = torch.nn.MSELoss()

losses, model_parameters_history = [], []


# MAIN ITERATION LOOP
def closure():
    optim.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, Y)
    loss.backward()
    print(loss.detach().cpu().numpy().copy(), model.model_params.detach().cpu().numpy().copy())
    losses.append(loss.detach().cpu().numpy().copy())
    model_parameters_history.append(model.model_params.detach().cpu().numpy().copy())
    return loss

loss = optim.step(closure)


# save the model parameters to a file
final_loss = losses[-1]
final_model_params = model_parameters_history[-1]


compression = "gzip"
with h5py.File(f"{MODEL_PARAMETERS_SAVE}.hdf5", "w") as w:
    w.create_dataset("final_loss",      data=np.array([final_loss]),      compression=compression)
    w.create_dataset("final_model_params", data=final_model_params, compression=compression)


