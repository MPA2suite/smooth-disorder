import numpy as np
import os
from ase.io import read
from scipy.constants import physical_constants

cell = read('POSCAR')
volume_A3 = cell.get_volume()
total_mass = cell.get_masses().sum()  # in atomic mass units

amu = physical_constants['atomic mass constant'][0]
conversion_factor = amu * 1e27

# converting from atomic mass units per angstrom^3 to kg/m^3
density_kg_m3 = total_mass / volume_A3 * conversion_factor * 1000

grid = [5, 5, 5]
num_q = int((grid[0] * grid[1] * grid[2] + 1) / 2)

weight = np.ones(num_q) * 2  # for the 5x5x5 grid, all q-points different from Gamma have weight 2
weight[0] = 1.0              # Gamma has weight 1

# load single results file written by 2a_convergence_serial.py
data = np.load('AFC_convergence_results.npz')
specific_heat_all        = data['specific_heat']          # (n_q, n_smear, n_temp)
tc_Allen_all             = data['tc_Allen']
tc_Allen_Hardy_all       = data['tc_Allen_Hardy']
tc_Allen_Lorentz_all     = data['tc_Allen_Lorentz']
tc_Allen_Lorentz_Hardy_all = data['tc_Allen_Lorentz_Hardy']
temperature_array        = data['temperatures']
smearings                = data['smearings']

n_smear = len(smearings)
n_temp  = len(temperature_array)

kappas_G       = np.zeros((n_smear, n_temp))
kappas_L       = np.zeros((n_smear, n_temp))
kappas_G_gamma = np.zeros((n_smear, n_temp))
kappas_L_gamma = np.zeros((n_smear, n_temp))

normalization_NV = grid[0] * grid[1] * grid[2] * volume_A3 * (1E-10 ** 3)

for i_smear in range(n_smear):
    specific_heat_T        = (specific_heat_all[:, i_smear, :]        * weight[:, None]).sum(axis=0)
    conductivity_Allen     = (tc_Allen_all[:, i_smear, :]             * weight[:, None]).sum(axis=0)
    conductivity_Allen_Hardy     = (tc_Allen_Hardy_all[:, i_smear, :] * weight[:, None]).sum(axis=0)
    conductivity_Allen_Lorentz   = (tc_Allen_Lorentz_all[:, i_smear, :] * weight[:, None]).sum(axis=0)
    conductivity_Allen_Lorentz_Hardy = (tc_Allen_Lorentz_Hardy_all[:, i_smear, :] * weight[:, None]).sum(axis=0)

    specific_heat_T              /= normalization_NV * density_kg_m3
    conductivity_Allen           /= normalization_NV
    conductivity_Allen_Hardy     /= normalization_NV
    conductivity_Allen_Lorentz   /= normalization_NV
    conductivity_Allen_Lorentz_Hardy /= normalization_NV

    kappas_G[i_smear]       = conductivity_Allen
    kappas_L[i_smear]       = conductivity_Allen_Lorentz
    kappas_G_gamma[i_smear] = tc_Allen_all[0, i_smear, :]       * weight[0] / (volume_A3 * (1E-10 ** 3))
    kappas_L_gamma[i_smear] = tc_Allen_Lorentz_all[0, i_smear, :] * weight[0] / (volume_A3 * (1E-10 ** 3))

os.makedirs('results', exist_ok=True)
np.save('./results/convergence_test.npy',
        np.array([kappas_G, kappas_L, kappas_G_gamma, kappas_L_gamma], dtype=object))
