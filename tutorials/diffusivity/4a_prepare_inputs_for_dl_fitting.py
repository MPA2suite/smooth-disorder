import string, re, struct, sys, math, os

import numpy as np
import pandas as pd

from ase.io import read, write
from typing import Tuple

import h5py
import matplotlib.pyplot as plt


# save the frequencies
def save_frequency_data_to_files(filename, frequencies, weights):
    compression="gzip"
    with h5py.File(f"{filename}.hdf5", "w") as w:
        w.create_dataset("frequencies", data=frequencies, compression=compression)
        w.create_dataset("weights", data=weights, compression=compression)


data_filename = "./data_save/IC_dataset_tensor.npz"
raw = np.load(data_filename)

save_filename = "./irg_t9_216_frequencies"
weights = raw['weights']
frequencies = raw['frequencies']
save_frequency_data_to_files(save_filename, frequencies, weights)



# calculate the diffusivity as a function of frequency and save it
def diffusivity_w(frequencies: np.ndarray, weights: np.ndarray, gamma_min_plateau: float,
                 diffusivities: np.ndarray, structure_file) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Computes diffusivity as function of frequency

    """

    width_frequency = gamma_min_plateau / 10
    min_frequency = sorted(frequencies.flatten())[3]
    max_frequency = np.amax(frequencies)
    sigma_gauss = np.sqrt(np.pi / 2.0) * gamma_min_plateau  # in cmm^-1
    num_q = weights.sum()

    frequencies_plot = np.arange(min_frequency - sigma_gauss, max_frequency + sigma_gauss, width_frequency)
    counts_all = np.zeros((len(frequencies_plot),))
    diffusivities_all = np.zeros((len(frequencies_plot,)))

    def gaussian(x, sigma=sigma_gauss):
        result = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (x / sigma) ** 2)
        return np.where(np.abs(x) > 2.5 * sigma, 0, result)

    
    # MAIN calculation
    for iq in range(len(weights)):
        weight = weights[iq]
        freq_modes = frequencies[iq]
        diffusivity = diffusivities[iq]

        for idx_bin in range(len(frequencies_plot)):
            frequency = frequencies_plot[idx_bin]

            bool_condition2 = np.abs(freq_modes - frequency) < 2.5 * sigma_gauss  # for frequency


            # shape: (num_modes close to frequency,)
            gauss_distr = gaussian(freq_modes[bool_condition2] - frequency)

            counts_all[idx_bin] += (weight * gauss_distr).sum()
            diffusivities_all[idx_bin] += (diffusivity[bool_condition2] * weight * gauss_distr).sum()

    cell = read(structure_file)
    volume_A3 = cell.get_volume()
    
    diffusivities_all = np.where(np.isclose(counts_all, 0.0), 0.0, diffusivities_all/counts_all)

    return counts_all / (volume_A3 * num_q), diffusivities_all, frequencies_plot


def obtain_gamma_plateau(infile: str = "data_save/IC_dataset.npz") -> float:
    """
    Returns gamma_min_plateau used to obtain quantities in the dataset
    :param infile: Location of the dataset
    :return: gamma_plateau
    """
    raw = np.load(infile)
    gamma_plateau = float(np.copy(raw['gamma_min_plateau']))
    return gamma_plateau



# read data
source = f"./data_save/IC_dataset_tensor.npz"
IC_dataset = np.load(source)

temperatures = np.copy(IC_dataset['temperatures'])
weights = np.copy(IC_dataset['weights'])
freq = np.copy(IC_dataset['frequencies'])

density = np.copy(IC_dataset['density'])

diffusivity_Allen = np.copy(raw['diff_Allen'])
diffusivity_Allen = np.diagonal(diffusivity_Allen[:, :, :, :], axis1=2, axis2=3).sum(axis=2) / 3

atomic_file = f"./POSCAR"
gamma_plateau = obtain_gamma_plateau(source)

# diffusivity as a function of frequency
_, diffusivity_plot, frequencies_plot = diffusivity_w(freq, weights, gamma_plateau, diffusivity_Allen, atomic_file)



# save data
save_dict = {'frequencies': frequencies_plot, 'diffusivities': diffusivity_plot, 'gamma_plateau': gamma_plateau}

cmd = f"mkdir -p ./data_save/diffusivity_w"
os.system(cmd)

outfile = f'./data_save/diffusivity_w/diffusivity_Allen_eta_conv.npz'
np.savez(outfile, **save_dict)


raw = np.load('./data_save/diffusivity_w/diffusivity_Allen_eta_conv.npz')
frequencies_diffusivity = raw['frequencies']
diffusivities = raw['diffusivities']

compression="gzip"
with h5py.File(f"irg_t9_216_diffusivity.hdf5", "w") as w:
    w.create_dataset("diffusivity", data=diffusivities, compression=compression)
    w.create_dataset("frequencies_plot", data=frequencies_diffusivity, compression=compression)


plt.figure(figsize=(16, 8))

plt.plot(frequencies_diffusivity, diffusivities)
plt.xlabel('Frequencies [cm^{-1}]')
plt.ylabel(r'Diffusivity [m^2 / s]')

plt.savefig("./diffusivity_plot.png", dpi=300)