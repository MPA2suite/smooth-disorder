import numpy as np
import os
from ase.io import read

gamma_min_plateau_cmm1 = 8.0

grid = [5, 5, 5]
num_q = int((grid[0] * grid[1] * grid[2] + 1) / 2)

cell = read('POSCAR')
n_atoms_structure = len(cell.get_positions())
num_modes = n_atoms_structure * 3
amount_of_modes = 1000
num_starts = np.arange(0, num_modes + amount_of_modes, amount_of_modes)
num_starts[-1] = num_modes  # to make sure we don't go over the array

for iq in range(0, num_q):
    for idx in range(len(num_starts) - 1):
        num_start = num_starts[idx]
        num_stop = num_starts[idx + 1]

        cmd = f"python 3a_tensor_conductivity_save.py {iq} {num_start} {num_stop} {gamma_min_plateau_cmm1}"
        print(cmd)
        os.system(cmd)
