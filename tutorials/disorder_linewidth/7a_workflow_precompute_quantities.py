import os
import sys

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ase
from ase.io import read

from tqdm import tqdm


from phonopy import Phonopy
from phonopy import file_IO as phonopy_file_IO

from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections


from smooth_disorder.structural import obtain_density, THzToCm, THz, Angstrom
from smooth_disorder.vis.interactive import *

from smooth_disorder.disorder_linewidth import run_band_structure_manual
from smooth_disorder.disorder_linewidth import run_phonon_mesh, save_mesh_data_to_files

from smooth_disorder.disorder_linewidth import lorentzian_numpy, flatten_arrays_freq_only
from smooth_disorder.disorder_linewidth import calculate_vdos_with_frequency
from smooth_disorder.disorder_linewidth import save_vdos_data_to_files

from smooth_disorder.disorder_linewidth import flatten_arrays
from smooth_disorder.disorder_linewidth import calculate_vdos_and_average_speed_with_frequency
from smooth_disorder.disorder_linewidth import save_vdos_speed_data_to_files



CRYSTAL_POSCAR   = "./1_graphite/POSCAR"
CRYSTAL_FC2      = "./1_graphite/fc2.hdf5"
SUPERCELL_MATRIX = np.diag([8, 8, 2])
MESH             = [128, 128, 32]
GAMMA_CENTER     = True

WORK_DIR = "./dl_workflow"
os.makedirs(WORK_DIR, exist_ok=True)

MESH_SAVE        = f"{WORK_DIR}/mesh_data"
CRYSTAL_VEL_SAVE      = f"{WORK_DIR}/crystal_vdos_group_vel"
SHIFTED_SAVE          = f"{WORK_DIR}/reduced_density_crystal_vdos_group_vel"


DISORDERED_POSCAR      = "./2_irg_t2/irg_t2_14009.vasp"
DISORDERED_FREQUENCIES = "./2_irg_t2/irg_t2_frequencies.hdf5"
DISORDERED_DIFFUSIVITY = "./2_irg_t2/diffusivity.hdf5"

DISORDERED_VDOS_SAVE  = f"{WORK_DIR}/disordered_vdos"

# Lorentzian half-width η for VDOS broadening [cm⁻¹].
# Controls spectral resolution — value used for graphite: 0.6 cm⁻¹.
GAMMA_BROADENING = 0.6



################################
# BAND STRUCTURE VISUALIZATION #
################################

# Special BZ points for hexagonal graphite — fractional reciprocal-lattice coordinates
atoms_ase = ase.io.read(CRYSTAL_POSCAR)
print(f"Special points in the BZ")
print(ase.dft.kpoints.get_special_points(atoms_ase.cell))

# Preferred path chosen from the BZ
path = [[[0, 0, 0], [0. , 0. , 0.5], [0.5, 0. , 0.5], [0.5, 0. , 0. ],
         [0., 0., 0.], [0.33333333, 0.33333333, 0.], [0.33333333, 0.33333333, 0.5]]]
labels = ["$\\Gamma$", "A", "L", "M", "$\\Gamma$", "K", "H"]

print(f"Running the band structure calculation")
frequencies, distances, qpoints = run_band_structure_manual(CRYSTAL_POSCAR, CRYSTAL_FC2, SUPERCELL_MATRIX, path, labels)

num_paths = len(distances)
num_modes = frequencies[0].shape[1]


# visualization
plt.figure(figsize=(16, 8))

for path in range(num_paths):
    for mode in range(num_modes):
        plt.plot(distances[path], frequencies[path][:, mode]*THzToCm, color=Colors[3])

y_min, y_max = 0, 1600
plt.vlines(0.0, y_min, y_max, color="black", lw=1)
plt.text(0.0-0.003, -50, labels[0])

for path in range(num_paths):
    plt.vlines(distances[path][-1], y_min, y_max, color="black", lw=1)
    plt.text(distances[path][-1]-0.003, -50, labels[path+1])

plt.ylabel("Frequencies [cm-1]")
plt.xlabel("Brillouin zone special points")

plt.xlim([-0.01, np.max(distances)+0.01])
plt.xticks([])
plt.savefig(f"{WORK_DIR}/crystal_band_structure.png", dpi=300)


################################################################
# OBTAIN FREQUENCIES AND GROUP VELOCITIES IN REFERENCE CRYSTAL #
################################################################


print(f"Obtaining the crystal group velocities...")
mesh_dict = run_phonon_mesh(CRYSTAL_POSCAR, CRYSTAL_FC2, SUPERCELL_MATRIX, MESH, GAMMA_CENTER)

save_mesh_data_to_files(MESH_SAVE,
                        mesh_dict['frequencies_cm'],
                        mesh_dict['weights'],
                        mesh_dict['qpoints'],
                        mesh_dict['group_velocities_ms'])

with h5py.File(f"{MESH_SAVE}.hdf5", "r") as f:
    frequencies_cm      = np.asarray(f["frequencies_cm"])       # (N_qpts, N_bands) [cm⁻¹]
    weights             = np.asarray(f["weights"])              # (N_qpts,)
    qpoints             = np.asarray(f["qpoints"])              # (N_qpts, 3) fractional
    group_velocities_ms = np.asarray(f["group_velocities_ms"])  # (N_qpts, N_bands, 3) [m/s]


plt.figure(figsize=(16, 8))

speed_2d = np.sqrt(
    np.square(group_velocities_ms).sum(axis=2) / 3
)

plt.scatter(frequencies_cm, speed_2d, color=Colors[3], s=0.01)


plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Group Velocity [m/s]")

plt.title("Group Velocity of reference crystal")

plt.savefig(f"{WORK_DIR}/crystal_group_velocities.png", dpi=300)


#############################################
# OBTAIN VDOS AND v(w) IN REFERENCE CRYSTAL #
#############################################

print(f"Obtaining the crystal VDOS")

frequencies_flat, weights_flat, speed_flat, weights_sum = flatten_arrays(
    frequencies_cm,
    weights,
    group_velocities_ms)

vdos_crystal, speed_crystal, freq_crystal = calculate_vdos_and_average_speed_with_frequency(
    frequencies_flat,
    weights_flat,
    speed_flat,
    GAMMA_BROADENING,
    CRYSTAL_POSCAR,
    weights_sum
)

save_vdos_speed_data_to_files(CRYSTAL_VEL_SAVE, freq_crystal, vdos_crystal, speed_crystal)


plt.figure(figsize=(16, 8))

plt.plot(freq_crystal, vdos_crystal, color=Colors[3])

plt.xlabel(r"Frequency [cm$^{-1}$]")
plt.ylabel(r"VDOS [THz$^{-1}$ nm$^{-3}$]")

plt.title("Vibrational Density of States of reference crystal")

plt.savefig(f"{WORK_DIR}/crystal_vdos.png", dpi=300)



plt.figure(figsize=(16, 8))

speed_2d = np.sqrt(
    np.square(group_velocities_ms).sum(axis=2) / 3
)

plt.scatter(frequencies_cm, speed_2d, color=Colors[0], s=0.01)

# the average of the group velocities at a given frequency
plt.plot(freq_crystal, speed_crystal, color=Colors[3])

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Group Velocity [m/s]")

plt.title("Group Velocity of reference crystal")

plt.savefig(f"{WORK_DIR}/crystal_group_velocities_vs_frequency.png", dpi=300)



##############################################################
# VISUALIZE FREQUENCIES AND DIFFUSIVITY IN DISORDERED SYSTEM #
##############################################################


with h5py.File(DISORDERED_FREQUENCIES, "r") as f:
    print(f.keys())
    frequencies = np.asarray(f['frequencies'])  # these are already saved in cm^-1
    weights = np.asarray(f['weights'])

print(f"Obtaining VDOS of the disordered system")
frequencies_flat, weights_flat, weights_sum = flatten_arrays_freq_only(frequencies, weights)

vdos_disordered, freq_disordered = calculate_vdos_with_frequency(
    frequencies_flat,
    weights_flat,
    GAMMA_BROADENING,
    DISORDERED_POSCAR,
    weights_sum
)

save_vdos_data_to_files(DISORDERED_VDOS_SAVE, freq_disordered, vdos_disordered)




plt.figure(figsize=(16, 8))

plt.plot(freq_disordered, vdos_disordered, color=Colors[0])

plt.xlabel("Frequency [cm$^{-1}$]")
plt.ylabel("VDOS [THz$^{-1}$ nm$^{-3}$]")

plt.title("Vibrational Density of States of the disordered system")

plt.xticks(np.arange(0, 1800, 100))

plt.savefig(f"{WORK_DIR}/disordered_system_vdos.png", dpi=300)



plt.figure(figsize=(16, 8))

plt.plot(freq_crystal, vdos_crystal, color=Colors[3], label="reference crystal")
plt.plot(freq_disordered, vdos_disordered, color=Colors[0], label='irradiated graphite')

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("VDOS [THz^-1 nm^-3]")

plt.title("Vibrational Densities of States of reference crystal and the disordered system")

plt.xticks(np.arange(0, 2000, 100))

plt.legend(loc='upper left')

plt.savefig(f"{WORK_DIR}/crystal_vs_disordered_system_vdos.png", dpi=300)





# read in the diffusivity
with h5py.File(DISORDERED_DIFFUSIVITY, "r") as f:
    print(f.keys())
    diffusivity_plot = np.asarray(f["diffusivity"])  # in m^2/s
    frequencies_plot_diff = np.asarray(f["frequencies_plot"])  # in cm^-1




plt.figure(figsize=(16, 8))

plt.plot(frequencies_plot_diff, diffusivity_plot*1e6, color=Colors[0])

plt.xlabel("Frequency [cm$^{-1}$]")
plt.ylabel("Diffusivity [mm$^2$/s]")

plt.title("Diffusivity of modes in the disordered system")

plt.xticks(np.arange(0, 1800, 100))

plt.savefig(f"{WORK_DIR}/disordered_system_diffusivity.png", dpi=300)



################################################################
# INFLUENCE OF DENSITY ON FREQUENCY SHIFTS IN THE CRYSTAL VDOS #
################################################################


density_crystal = obtain_density(read(CRYSTAL_POSCAR))
density_disordered = obtain_density(read(DISORDERED_POSCAR))

density_factor = np.power(density_disordered/density_crystal, 1/3)
print(f"  ρ_crystal    = {density_crystal:.4f} g/cm³")
print(f"  ρ_disordered = {density_disordered:.4f} g/cm³")
print(f"  Density factor = (ρ_dis/ρ_crys)^(1/3) = {density_factor:.4f}")



# load the mesh data
with h5py.File(f"{MESH_SAVE}.hdf5", "r") as f:
    frequencies_cm      = np.asarray(f["frequencies_cm"])       # (N_qpts, N_bands) [cm⁻¹]
    weights             = np.asarray(f["weights"])              # (N_qpts,)
    qpoints             = np.asarray(f["qpoints"])              # (N_qpts, 3) fractional
    group_velocities_ms = np.asarray(f["group_velocities_ms"])  # (N_qpts, N_bands, 3) [m/s]


print(f"Applying frequency shifts and recalculating crystal VDOS")

# flatten the arrays for easier VDOS calculation
frequencies_flat, weights_flat, speed_flat, weights_sum = flatten_arrays(
    frequencies_cm,
    weights,
    group_velocities_ms)

# we shift the frequencies by the multiplicative factor
shifted_frequencies_flat = frequencies_flat * density_factor

shifted_vdos_crystal, shifted_speed_crystal, shifted_freq_crystal = calculate_vdos_and_average_speed_with_frequency(
    shifted_frequencies_flat,
    weights_flat,
    speed_flat,
    GAMMA_BROADENING,
    CRYSTAL_POSCAR,
    weights_sum
)

save_vdos_speed_data_to_files(SHIFTED_SAVE, shifted_freq_crystal, shifted_vdos_crystal, shifted_speed_crystal)



plt.figure(figsize=(16, 8))

plt.plot(shifted_freq_crystal, shifted_vdos_crystal, color=Colors[3])

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("VDOS [THz^-1 nm^-3]")

plt.title("shifted Vibrational Density of States of the reference crystal")

plt.savefig(f"{WORK_DIR}/crystal_shifted_vdos.png", dpi=300)


plt.figure(figsize=(16, 8))

plt.scatter(shifted_frequencies_flat, speed_flat, color=Colors[0], s=0.01)
plt.plot(shifted_freq_crystal, shifted_speed_crystal, color=Colors[3])

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Group Velocity [m/s]")

plt.title("Group Velocity of the reference crystal with shifted frequencies")

plt.savefig(f"{WORK_DIR}/crystal_shifted_group_velocities.png", dpi=300)

