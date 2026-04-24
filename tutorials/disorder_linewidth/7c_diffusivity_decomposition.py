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

from smooth_disorder.disorder_linewidth import lorentzian_numpy, prepare_fitting_inputs
from smooth_disorder.disorder_linewidth import evaluate_linewidth_and_model_prediction


CRYSTAL_POSCAR    = "./1_graphite/POSCAR"
DISORDERED_POSCAR = "./2_irg_t2/irg_t2_14009.vasp"
DISORDERED_DIFFUSIVITY = "./2_irg_t2/diffusivity.hdf5"

WORK_DIR = "./dl_workflow"

CRYSTAL_VEL_SAVE     = f"{WORK_DIR}/crystal_vdos_group_vel"
DISORDERED_VDOS_SAVE = f"{WORK_DIR}/disordered_vdos"
SHIFTED_SAVE         = f"{WORK_DIR}/reduced_density_crystal_vdos_group_vel"

MODEL_PARAMETERS_SAVE = f"{WORK_DIR}/model_parameters"


# read in the preprocessed data
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

# load the model parameters
with h5py.File(f"{MODEL_PARAMETERS_SAVE}.hdf5", "r") as f:
    model_params = np.asarray(f["final_model_params"])
    L_ref, R_ref = model_params[0]*1e1, model_params[1]*1e-6  # convert to standard units

print(f"Grain boundary length: {L_ref} Angstrom")
print(f"Defect scattering parameter: {R_ref} THz cm nm^3")


# find the PDC VDOS
(
    vdos_PDC,
    disorder_linewidth,
    defect_linewidth,
    Casimir_model_linewidth
) = evaluate_linewidth_and_model_prediction(
        density_crystal,
        density_disordered,
        freq_disordered,
        vdos_disordered,
        interp_shifted_freq_crystal,
        interp_shifted_vdos_crystal,
        interp_shifted_speed_crystal,
        L=L_ref,
        R=R_ref,
    )


###########################
# DISORDER LINEWIDTH PLOT #
###########################


plt.figure(figsize=(16, 8))

plt.plot(interp_shifted_freq_crystal, defect_linewidth, color=Colors[3], label="defect contribution")
plt.plot(interp_shifted_freq_crystal, Casimir_model_linewidth, color=Colors[0], label="grain boundary contribution")
plt.plot(interp_shifted_freq_crystal, disorder_linewidth, color=Colors[2], label="total disorder linewidth")

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Disorder linewidth [cm^-1]")

plt.title("Disorder linewidth and its contributions")

plt.legend(loc="upper left")

plt.savefig(f"{WORK_DIR}/disorder_linewidth.png", dpi=300)



########################
# VDOS COMPARISON PLOT #
########################



def obtain_crystal_vdos():

    with h5py.File(f"{CRYSTAL_VEL_SAVE}.hdf5", "r") as f:
        freq_crystal = np.asarray(f["frequencies_bin"])  # [cm⁻¹]
        vdos_crystal = np.asarray(f["vdos_return"])      # [THz⁻¹ nm⁻³]

    return freq_crystal, vdos_crystal

freq_crystal, vdos_crystal = obtain_crystal_vdos()



plt.figure(figsize=(16, 8))

plt.plot(freq_disordered, vdos_disordered, color=Colors[0], label='disordered system')
plt.plot(freq_disordered, vdos_PDC, color=Colors[1], label='PDC model of the VDOS of disordered system')


plt.xlabel("Frequency [cm^-1]")
plt.ylabel("VDOS [THz^-1 nm^-3]")

plt.title("VDOS of disordered system vs PDC VDOS model")

plt.legend(loc="upper left")

plt.savefig(f"{WORK_DIR}/disordered_vs_pdc_vdos.png", dpi=300)


#############################
# PROPAGATION VELOCITY PLOT #
#############################


with h5py.File(DISORDERED_DIFFUSIVITY, "r") as f:
    print(f.keys())
    diffusivity_plot = np.asarray(f["diffusivity"])  # in m^2/s
    frequencies_plot_diff = np.asarray(f["frequencies_plot"])  # in cm^-1


lifetime = 1/disorder_linewidth * THzToCm * 1e-12  # in seconds
extrapolation_value = 1/24 * THzToCm * 1e-12  # value for frequencies above maximum
lifetime_function = interp1d(interp_shifted_freq_crystal, lifetime, bounds_error=False, fill_value=extrapolation_value)
lifetime_interpolated = lifetime_function(frequencies_plot_diff)

plt.figure(figsize=(16, 8))

plt.plot(interp_shifted_freq_crystal, interp_shifted_speed_crystal * 1e-3, color=Colors[3], label='crystal graphite')
plt.plot(frequencies_plot_diff, np.sqrt(diffusivity_plot/lifetime_interpolated) * 1e-3, color=Colors[0], label='irradiated graphite')


plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Propagation velocity [km/s]")

plt.title("Propagation velocity in reference crystal and disordered system")

plt.legend(loc="upper right")

plt.savefig(f"{WORK_DIR}/propagation_velocity.png", dpi=300)


plt.figure(figsize=(16, 8))

plt.plot(frequencies_plot_diff, np.sqrt(diffusivity_plot*lifetime_interpolated) * 1e10)

plt.xlabel("Frequency [cm^-1]")
plt.ylabel("Mean free path [A]")

plt.title("Mean free path in the disordered system")

plt.savefig(f"{WORK_DIR}/mean_free_path.png", dpi=300)
