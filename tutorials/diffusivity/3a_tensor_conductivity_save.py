# import needed libraries
import numpy as np
import os, sys
import h5py
from typing import List

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

if len(sys.argv) > 1:
    iq, num_start, num_stop, gamma_min_plateau_cmm1 = (int(sys.argv[1]), int(sys.argv[2]),
                                                        int(sys.argv[3]), float(sys.argv[4]))
else:
    print('error: iq, num_start, num_stop, gamma_min_plateau_cmm1 not given')
    exit()

temperature_list = np.arange(50, 2001, 50).tolist()
n_temp = len(temperature_list)

gamma_min_plateau_Ry = gamma_min_plateau_cmm1 / rY_TO_CMM1
sigma_Gauss_Ry = np.sqrt(np.pi / 2.0) * gamma_min_plateau_Ry


# -------------------------------------------------------------
# define function for calculating the conductivity
def is_acoustic_mode_at_Gamma(iq, im):
    res = False
    if (iq == 0 and im < 3):
        res = True
    return res


def conductivity_q(iq: int = iq,
                   num_start: int = num_start,
                   num_stop: int = num_stop,
                   sigma_Gauss: float = sigma_Gauss_Ry,
                   temp_list: List = temperature_list):
    """
    Saves and returns specific heat, Allen conductivity and diffusivity, weight, temperatures and q-point.
    Quantities are resolved w.r.t. temperature and mode.

    :param iq: q-point for calculation
    :param num_start: mode to start - used for parallelisation
    :param num_stop: mode to stop (not inclusive) - used for parallelisation
    :param sigma_Gauss: sigma which broadens the delta function into a Gaussian [Rydberg]
    :param temp_list: list of temperatures for which to calculate conductivities

    :return: specific_heat_q,             shape: (num_stop-num_start, n_temp)
             thermal_conductivity_Allen_q, shape: (num_stop-num_start, n_temp, 3, 3)
             diffusivity_Allen_q,          shape: (num_stop-num_start, 3, 3)
             weight,
             np.asarray(temp_list),        shape: (n_temp,)
             iq
    """
    smearing = sigma_Gauss * rY_TO_CMM1 / np.sqrt(np.pi / 2.0)  # value of the eta parameter used here

    # read frequencies and velocities
    filename_vel = './velocity_operators/save_%d.hdf5' % (iq)
    f_vel = h5py.File(filename_vel, 'r')

    # read velocity operator in phonopy format and convert to QE atomic units
    array_V_operator = np.asarray(f_vel['velocity_operator'][:]) * (THz * Angstrom / (2.0 * np.pi * ryvel_si))
    array_frequency = np.asarray(f_vel['frequency'])
    weight = int(np.copy(f_vel['weight']))

    # read frequencies
    num_modes = len(array_frequency)
    n_temp = len(temp_list)

    # setting up arrays for thermal conductivity dataset
    array_modes = num_stop - num_start
    specific_heat_q = np.zeros((array_modes, n_temp))  # after conversion should be in SI units

    thermal_conductivity_Allen_q = np.zeros((array_modes, n_temp, 3, 3))
    diffusivity_Allen_q = np.zeros([array_modes, 3, 3])

    conv_diffusivity = ((tpi * ryvel_si) ** 2) * ryau_sec / tpi
    conv_spec_heat = (1.0 / k_BOLTZMANN_SI) * (rYDBERG_SI) ** 2

    # -----------------
    # MAIN FOR-LOOP
    # -----------------

    for id_temp in range(0, n_temp):
        temperature = temp_list[id_temp]
        tempm1 = 1. / (temperature * k_BOLTZMANN_RY)

        frequencies_Ry = (array_frequency * THzToCm) / rY_TO_CMM1
        bose_all = 1.0 / (np.exp(frequencies_Ry * tempm1) - 1.0)
        f_bose_all = bose_all * (bose_all + 1.0)
        spec_heat_all = (1.0 / (temperature ** 2)) * np.square(frequencies_Ry) * f_bose_all

        specific_heat_q[:, id_temp] = spec_heat_all[num_start:num_stop]

        for im1 in range(num_start, num_stop):  # loop maintained so the memory requirements for large models are not huge

            if is_acoustic_mode_at_Gamma(iq, im1):
                continue

            omega_1_Ry = frequencies_Ry[im1]
            spec_heat_s1 = spec_heat_all[im1]

            # vectorisation over second mode
            local_frequencies = frequencies_Ry
            local_spec_heat = spec_heat_all
            n_skip = 3 if iq == 0 else 0  # used for skipping zero modes
            local_velocity_operator = array_V_operator[im1]

            # distributions
            Gaussian = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_Gauss)) * np.exp(
                -0.5 * ((local_frequencies - omega_1_Ry) / sigma_Gauss) ** 2)

            # calculate v_alpha v_beta tensor
            local_velocity_tensor = np.zeros((3, 3, len(local_velocity_operator)))
            for alpha in range(3):
                for beta in range(3):
                    local_velocity_tensor[alpha, beta, :] = (local_velocity_operator[:,
                                                            alpha] * local_velocity_operator.conj()[:, beta]).real

            # prefactors
            prefactor_conductivity_tensor = (((omega_1_Ry + local_frequencies) / 4.0) *
                                      ((spec_heat_s1 / omega_1_Ry) + (local_spec_heat / local_frequencies)) *
                                      np.pi * local_velocity_tensor)

            prefactor_diffusivity_tensor = (
                    ((omega_1_Ry + local_frequencies) / (2.0 * (spec_heat_s1 + local_spec_heat))) *
                    ((spec_heat_s1 / omega_1_Ry) + (local_spec_heat / local_frequencies)) *
                    local_velocity_tensor *
                    np.pi)

            thermal_conductivity_Allen_q[im1 - num_start, id_temp, :, :] = (prefactor_conductivity_tensor * Gaussian)[:, :, n_skip:].sum(axis=2)

            # diffusivity is temperature-independent; compute once at the last temperature step
            if id_temp == n_temp - 1:
                diffusivity_Allen_q[im1 - num_start] = (prefactor_diffusivity_tensor * Gaussian)[:, :, n_skip:].sum(axis=2)

    result = (specific_heat_q * conv_spec_heat,
             thermal_conductivity_Allen_q * conv_spec_heat * conv_diffusivity,
             diffusivity_Allen_q * conv_diffusivity,
             weight,
             np.asarray(temp_list),
             iq)

    os.makedirs('data_save/interpolation_conductivity_dataset', exist_ok=True)
    np.save(
        f'./data_save/interpolation_conductivity_dataset/results_iq_{iq}_start_{num_start}_stop_{num_stop}_gamma_{int(np.round(smearing * 1000))}.npy',
        np.array(result, dtype=object))

    return result

result = conductivity_q()
