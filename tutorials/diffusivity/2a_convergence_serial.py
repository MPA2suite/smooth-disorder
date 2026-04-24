# import needed libraries
import numpy as np
import os, sys
import h5py
from ase.io import read
from typing import Tuple, List
import glob
from tqdm import tqdm

# CONSTANTS
num_modes_start = 0
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

temperature_list = [100, 300, 1500]


# -------------------------------------------------------------
# define function for calculating the conductivity
def is_acoustic_mode_at_Gamma(iq, im):
    res = False
    if (iq == 0 and im < 3):
        res = True
    return res


def conductivity_q(iq: int, sigma_Gauss: float, temp_list: List = temperature_list):
    """
    :param iq: q-point on a mesh for calculation of the conductivity
    :param sigma_Gauss: sigma which broadens the delta function into a Gaussian [Rydberg]
    :param temp_list: list of temperatures for which to calculate conductivities
    :return: tuple (specific_heat_q, tc_Allen_q, tc_Allen_Hardy_q,
                    tc_Allen_Lorentz_q, tc_Allen_Lorentz_Hardy_q, temp_array, iq)
    """
    smearing = sigma_Gauss * rY_TO_CMM1 / np.sqrt(np.pi / 2.0)

    filename_vel = './velocity_operators/save_%d.hdf5' % (iq)
    f_vel = h5py.File(filename_vel, 'r')

    # read velocity operator in phonopy format and convert to QE atomic units
    array_V_operator = np.asarray(f_vel['velocity_operator'][:]) * (THz * Angstrom / (2.0 * np.pi * ryvel_si))
    array_frequency = np.asarray(f_vel['frequency'])
    weight = int(np.copy(f_vel['weight']))

    shape_arr = array_V_operator.shape
    num_modes = shape_arr[1]
    n_temp = len(temp_list)

    # setting up arrays for thermal conductivity
    thermal_conductivity_Allen_q = np.zeros(n_temp)
    thermal_conductivity_Allen_q_Hardy = np.zeros(n_temp)

    thermal_conductivity_Allen_q_Lorentz = np.zeros(n_temp)
    thermal_conductivity_Allen_q_Lorentz_Hardy = np.zeros(n_temp)

    specific_heat_q = np.zeros(n_temp)

    lw_Ry_1_approx = sigma_Gauss / (np.sqrt(np.pi / 2.0))
    lw_Ry_2_approx = lw_Ry_1_approx

    conv_diffusivity = ((tpi * ryvel_si) ** 2) * ryau_sec / tpi
    conv_spec_heat = (1.0 / k_BOLTZMANN_SI) * (rYDBERG_SI) ** 2

    for id_temp in range(0, n_temp):
        temperature = temp_list[id_temp]
        tempm1 = 1. / (temperature * k_BOLTZMANN_RY)

        frequencies_Ry = (array_frequency * THzToCm) / rY_TO_CMM1
        bose_all = 1.0 / (np.exp(frequencies_Ry * tempm1) - 1.0)
        f_bose_all = bose_all * (bose_all + 1.0)
        spec_heat_all = (1.0 / (temperature ** 2)) * np.square(frequencies_Ry) * f_bose_all

        if iq == 0:
            specific_heat_q[id_temp] = spec_heat_all[3:].sum()
        else:
            specific_heat_q[id_temp] = spec_heat_all.sum()

        for im1 in range(0, num_modes):  # loop maintained so the memory requirements for large models are not huge

            if is_acoustic_mode_at_Gamma(iq, im1):
                continue

            omega_1_Ry = frequencies_Ry[im1]
            spec_heat_s1 = spec_heat_all[im1]

            # define Gaussian and Lorentzian distributions
            Gaussian = (1.0 / (np.sqrt(2.0 * np.pi) * sigma_Gauss)) * np.exp(
                -0.5 * ((frequencies_Ry - omega_1_Ry) / sigma_Gauss) ** 2)
            Lorenztian = (1.0 / np.pi) * ((0.5 * (lw_Ry_1_approx + lw_Ry_2_approx)) / (
                    (frequencies_Ry - omega_1_Ry) ** 2 + 0.25 * (lw_Ry_1_approx + lw_Ry_2_approx) ** 2))

            # calculate square mod average of velocities
            square_mod_avg_Ry = ((array_V_operator[im1] * np.conjugate(array_V_operator[im1])).sum(axis=1) / 3.0).real

            # used for conversion to Hardy conductivity
            factor_velocity = (frequencies_Ry + omega_1_Ry) / (2.0 * np.sqrt(frequencies_Ry * omega_1_Ry))
            factor_velocity_square = np.square(factor_velocity)

            # Allen Feldman conductivity with Gaussian + its Hardy counterpart
            contr_Allen = (
                    ((frequencies_Ry + omega_1_Ry) / 4.0) *
                    ((spec_heat_all / frequencies_Ry) + (spec_heat_s1 / omega_1_Ry)) *
                    square_mod_avg_Ry *
                    np.pi * Gaussian)

            if iq == 0:  # avoid the zero translation modes
                thermal_conductivity_Allen_q[id_temp] += contr_Allen[3:].sum()
                thermal_conductivity_Allen_q_Hardy[id_temp] += (contr_Allen[3:] * factor_velocity_square[3:]).sum()
            else:
                thermal_conductivity_Allen_q[id_temp] += contr_Allen.sum()
                thermal_conductivity_Allen_q_Hardy[id_temp] += (contr_Allen * factor_velocity_square).sum()

            # Allen and Feldman conductivity with Lorentzian function + Hardy counterpart
            contr_Lorentz = (
                    ((frequencies_Ry + omega_1_Ry) / 4.0) *
                    ((spec_heat_all / frequencies_Ry) + (spec_heat_s1 / omega_1_Ry)) *
                    square_mod_avg_Ry *
                    np.pi * Lorenztian)

            if iq == 0:
                thermal_conductivity_Allen_q_Lorentz[id_temp] += contr_Lorentz[3:].sum()
                thermal_conductivity_Allen_q_Lorentz_Hardy[id_temp] += (
                        contr_Lorentz[3:] * factor_velocity_square[3:]).sum()
            else:
                thermal_conductivity_Allen_q_Lorentz[id_temp] += contr_Lorentz.sum()
                thermal_conductivity_Allen_q_Lorentz_Hardy[id_temp] += (contr_Lorentz * factor_velocity_square).sum()

    return (
        specific_heat_q * conv_spec_heat,
        thermal_conductivity_Allen_q * conv_spec_heat * conv_diffusivity,
        thermal_conductivity_Allen_q_Hardy * conv_spec_heat * conv_diffusivity,
        thermal_conductivity_Allen_q_Lorentz * conv_spec_heat * conv_diffusivity,
        thermal_conductivity_Allen_q_Lorentz_Hardy * conv_spec_heat * conv_diffusivity,
        np.asarray(temp_list),
        iq)


num_q = len(glob.glob("velocity_operators/save*.hdf5"))
list_smear = [0.025, 0.05, 0.075, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 30, 40, 60, 80, 100]
n_smear = len(list_smear)
n_temp = len(temperature_list)

# preallocate result arrays indexed [iq, i_smear, i_temp]
specific_heat        = np.zeros((num_q, n_smear, n_temp))
tc_Allen             = np.zeros((num_q, n_smear, n_temp))
tc_Allen_Hardy       = np.zeros((num_q, n_smear, n_temp))
tc_Allen_Lorentz     = np.zeros((num_q, n_smear, n_temp))
tc_Allen_Lorentz_Hardy = np.zeros((num_q, n_smear, n_temp))

for iq in tqdm(range(num_q), desc='q-points'):
    for i_smear, smear in enumerate(list_smear):
        sigma_Gauss_Ry = smear * np.sqrt(np.pi / 2.0) / rY_TO_CMM1
        result = conductivity_q(iq=iq, sigma_Gauss=sigma_Gauss_Ry)
        specific_heat[iq, i_smear]          = result[0]
        tc_Allen[iq, i_smear]               = result[1]
        tc_Allen_Hardy[iq, i_smear]         = result[2]
        tc_Allen_Lorentz[iq, i_smear]       = result[3]
        tc_Allen_Lorentz_Hardy[iq, i_smear] = result[4]

np.savez('AFC_convergence_results.npz',
         specific_heat=specific_heat,
         tc_Allen=tc_Allen,
         tc_Allen_Hardy=tc_Allen_Hardy,
         tc_Allen_Lorentz=tc_Allen_Lorentz,
         tc_Allen_Lorentz_Hardy=tc_Allen_Lorentz_Hardy,
         temperatures=np.array(temperature_list),
         smearings=np.array(list_smear))
