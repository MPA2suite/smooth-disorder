#!/usr/bin/env python

import sys, os
import time
import numpy as np
import h5py
from tqdm import tqdm

import phonopy
from phonopy import Phonopy
from phonopy import file_IO
from phonopy.interface.calculator import read_crystal_structure
VaspToTHz = phonopy.physical_units.get_physical_units().DefaultToTHz

from wte.velocity_operator import VelocityOperator

from ase.io import read
from ase.units import J, _hplanck  # conversion factors
from ase.dft.kpoints import bandpath


global start_time

# if len(sys.argv) > 1:
#     iq = int(sys.argv[1])
# else:
#     print('error: iq not given')
#     exit()

withBORN = False


def write__to_hdf5(filename, frequencies, qpoints, weights, velocity_operators):
    compression = "gzip"
    with h5py.File(filename, "w") as w:
        w.create_dataset("frequency", data=frequencies, compression=compression)
        w.create_dataset("qpoint", data=qpoints)
        w.create_dataset("weight", data=weights)
        w.create_dataset("velocity_operator", data=velocity_operators, compression=compression)



start_time = time.time()
eV2Hz = 1 / (J * _hplanck)

# input
primitive_filename = "POSCAR"
supercell = np.diag((2, 2, 2))
mesh = [5, 5, 5]
interpolation_mesh_size = mesh[0]

atoms, string = read_crystal_structure(filename=primitive_filename)
at_positions = atoms.scaled_positions
nat = len(at_positions)
nat3 = 3 * nat

print(nat3)


# setup FC2
phonons = Phonopy(atoms, supercell)

force_constants = file_IO.read_force_constants_hdf5(filename="fc2.hdf5")
primitive = phonons.primitive

if withBORN:
    nac_params = file_IO.parse_BORN(primitive, filename="BORN")
    phonons.nac_params = nac_params

phonons.force_constants = force_constants
phonons.symmetrize_force_constants()



# diagonalize the dynamical matrix
phonons.run_mesh(mesh, is_gamma_center=True, with_group_velocities=False)
mesh_dict = phonons.get_mesh_dict()
qpoints = mesh_dict['qpoints']
weights = mesh_dict['weights']
frequencies = np.asarray(mesh_dict['frequencies'])
n_q_pt = len(qpoints)
time_absolute_1 = time.time()
partial_time = time.time() - start_time

print('time 1=', partial_time)



# calculation of vel op elements
os.makedirs('velocity_operators', exist_ok=True)
for iq in tqdm(range(n_q_pt), desc='q-points'):
    q_pt = qpoints[iq, :]
    vel_op = VelocityOperator(
        dynamical_matrix=phonons.dynamical_matrix,
        q_length=5e-6,
        symmetry=phonons.primitive_symmetry,
        frequency_factor_to_THz=VaspToTHz,
    )
    vel_op.run([q_pt])
    write__to_hdf5('velocity_operators/save_%d.hdf5' % (iq), frequencies[iq, :], qpoints[iq, :], weights[iq], vel_op.velocity_operators[0, :, :, :])


print('max_f [cm^-1]=', np.amax(frequencies) * 33.356)
print('min_f [cm^-1]=', np.amin(frequencies) * 33.356)

time_absolute_2 = time.time()
total_time = time_absolute_2 - start_time

print('diagonalization time=', time_absolute_2 - time_absolute_1)
print('total time=', total_time)