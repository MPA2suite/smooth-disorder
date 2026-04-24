import os, sys
import numpy as np
from ase.io import read
from scipy.constants import physical_constants
import h5py
import glob


# CONSTANTS
rY_TO_CMM1 = 109737.315701113
hARTREE_SI = 4.35974394E-18
rYDBERG_SI = hARTREE_SI / 2.0
H_PLANCK_SI = 6.62606896E-34
ryau_sec = H_PLANCK_SI / rYDBERG_SI
k_BOLTZMANN_SI = 1.3806504E-23  # J K^-1
aU_SEC = H_PLANCK_SI / (2. * np.pi) / hARTREE_SI
aU_TERAHERTZ = aU_SEC * 1.0E+12
rY_TO_THZ = 1.0 / aU_TERAHERTZ / (4 * np.pi)
k_BOLTZMANN_RY = k_BOLTZMANN_SI / rYDBERG_SI
BOHR_TO_M = 0.52917720859E-10
ryvel_si = BOHR_TO_M / ryau_sec
SpeedOfLight = 299792458  # [m/s]
THzToCm = 1.0e12 / (SpeedOfLight * 100)  # [cm^-1] 33.356410
Angstrom = 1.0e-10  # [m]
THz = 1.0e12  # [/s]
tpi = 2.0 * np.pi
eV_TO_JOULE = 1.602177e-19
amu_TO_kg = 1.660539e-27

temp_list = np.arange(50, 2001, 50).tolist()
n_temp = len(temp_list)


# change parameters HERE
grid = [5, 5, 5]
num_q = int((grid[0] * grid[1] * grid[2] + 1) / 2)
gamma_min_plateau_cmm1 = 8.0



# get density
cell = read('POSCAR')
n_atoms_structure = len(cell.get_positions())
num_modes = n_atoms_structure * 3
amount_of_modes = 1000
num_starts = np.arange(0, num_modes + amount_of_modes, amount_of_modes)
num_starts[-1] = num_modes


volume_A3 = cell.get_volume()
total_mass = cell.get_masses().sum()  # in atomic mass units

amu = physical_constants['atomic mass constant'][0]
conversion_factor = amu * 1e27

# converting from atomic mass units per angstrom^3 to kg/m^3
density_kg_m3 = total_mass / volume_A3 * conversion_factor * 1000

dataset = {'name': f'density {density_kg_m3:.3f}, {n_atoms_structure} atoms', 'weights': [], 'qpoints': [],
           'frequencies': [], 'temperatures': temp_list, 'specific_heats': [],
           'tc_Allen': [], 'diff_Allen': [],
           'gamma_min_plateau': gamma_min_plateau_cmm1, 'density': density_kg_m3}

for iq in range(num_q):
    file = f"velocity_operators/save_{iq}.hdf5"

    with h5py.File(file, 'r') as f:
        keys = list(f.keys())
        print(f"file: {file}, keys: {keys}")
        dataset["frequencies"].append(np.copy(f['frequency'])[:] * THzToCm)

        dataset["weights"].append(int(np.copy(f['weight'])))
        dataset['qpoints'].append(np.copy(f['qpoint']))

    temp = {'specific_heats': [],
            'tc_Allen': [],
            'diff_Allen': []}

    for idx in range(len(num_starts) - 1):
        num_start = num_starts[idx]
        num_stop = num_starts[idx + 1]
        infile = f'./data_save/interpolation_conductivity_dataset/results_iq_{iq}_start_{num_start}_stop_{num_stop}_gamma_{int(1000*gamma_min_plateau_cmm1)}.npy'
        results = np.load(infile, allow_pickle=True)

        normalization_NV = grid[0] * grid[1] * grid[2] * volume_A3 * (1E-10 ** 3)

        temp['specific_heats'].append(results[0] / (normalization_NV * density_kg_m3))
        temp['tc_Allen'].append(results[1] / normalization_NV)
        temp['diff_Allen'].append(results[2])


    for key in ['specific_heats', 'tc_Allen', 'diff_Allen']:
        # perform vstack
        curr_array = temp[key]
        n_stacks = len(curr_array)
        temp[key] = np.vstack([curr_array[local_idx] for local_idx in range(n_stacks)])

        dataset[key].append(temp[key])

list_keys = ['weights', 'qpoints', 'frequencies', 'specific_heats', 'temperatures',
             'tc_Allen', 'diff_Allen']

for key in list_keys:
    dataset[key] = np.array(dataset[key])

outfile = f'data_save/IC_dataset_tensor.npz'
np.savez(outfile, **dataset)