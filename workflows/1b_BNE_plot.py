import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from ase.io import read
from scipy.interpolate import interp1d

from tqdm import tqdm

from smooth_disorder.vis.interactive import *

from config import BNE_WORK_DIR, BNE_FOLDER, STRUCTURE_IDX, LOCAL_ENVIRONMENT_NAT, N_START, N_STOP


BNE_data_directory = f"{BNE_WORK_DIR}/{BNE_FOLDER}"


BNE_data = {}

structure_idx = STRUCTURE_IDX

temp_BNE = []
temp_nat = []
for LE_nat in LOCAL_ENVIRONMENT_NAT:
    with h5py.File(f"{BNE_data_directory}/structure_{structure_idx}/entropy_number_{LE_nat}.hdf5", "r") as f:
        temp_BNE.append(np.asarray(f['entropy'])[0])
        temp_nat.append(np.asarray(f['number_of_atoms'])[0])

temp_BNE, temp_nat = np.array(temp_BNE), np.array(temp_nat)

BNE_data[f"{structure_idx}_BNE"] = temp_BNE
BNE_data[f"{structure_idx}_nat"] = temp_nat




n_start, n_stop = N_START, N_STOP

def normalised_BNE(n_atoms, entropy, n_start=n_start, n_stop=n_stop):
    """
    Estimate the BNE growth rate by averaging BNE(n)/n over [n_start, n_stop].

    Returns
    -------
    BNE_mean : float
        Mean BNE(n)/n in nats per atom — the growth rate estimate.
    BNE_std : float
        Standard deviation of BNE(n)/n over the window.
    """
    idx_start = np.arange(0, len(n_atoms), 1)[n_atoms == n_start][-1]
    idx_stop  = np.arange(0, len(n_atoms), 1)[n_atoms == n_stop][0]

    local_n_atoms  = n_atoms[idx_start:idx_stop+1]
    local_quotient = entropy[idx_start:idx_stop+1] / local_n_atoms

    BNE_mean = np.mean(local_quotient)
    BNE_std  = np.std(local_quotient, ddof=1)

    return BNE_mean, BNE_std



structure_idx = 0
nat  = BNE_data[f"{structure_idx}_nat"]
bne  = BNE_data[f"{structure_idx}_BNE"]


bne_mean, bne_std = normalised_BNE(nat, bne)
print(f"Growth rate (BNE/n averaged over n={n_start}–{n_stop}): {bne_mean:.4f} ± {bne_std:.4f} nats/atom")


plt.figure(figsize=(16, 8))
plt.plot(nat, bne, color=Colors[0], label="BNE (disordered system)")
plt.ylabel("BNE (nats)")
plt.xlabel("LAE size $n$ (number of atoms)")

plt.text(15, 2.3, f"Growth rate: {bne_mean:.4f} ± {bne_std:.4f}", fontsize=14)

plt.xlim([0, 40])
plt.legend()
plt.title("BNE vs local environment size")
plt.savefig(f"{BNE_data_directory}/structure_{structure_idx}/bne_growth.png", dpi=300)

