import h5py
import numpy as np

from ase.io import read
from scipy.interpolate import interp1d

from tqdm import tqdm

from smooth_disorder.structural import obtain_density, THzToCm, THz, Angstrom

from phonopy import Phonopy
from phonopy import file_IO as phonopy_file_IO

from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

import torch


def run_band_structure_manual(
    poscar_path: str,
    fc2_path: str,
    supercell_matrix: np.ndarray,
    path: list,
    labels: list,
    npoints: int = 51,
):
    """Run a Phonopy band-structure calculation along a user-supplied BZ path.

    Parameters
    ----------
    poscar_path : str
        Path to the POSCAR file (primitive cell).
    fc2_path : str
        Path to the second-order force constants HDF5 file.
    supercell_matrix : np.ndarray, shape (3, 3)
        Supercell transformation matrix used when computing the force constants.
    path : list of list of list of float
        List of q-point segment endpoints in reduced coordinates,
        e.g. ``[[[0,0,0],[0.5,0,0],[0.5,0,0.5]]]``.
    labels : list of str
        High-symmetry point labels for each segment endpoint,
        e.g. ``['Γ', 'M', 'L']``.
    npoints : int, optional
        Number of q-points per path segment (default 51).

    Returns
    -------
    frequencies : list of np.ndarray
        Frequencies in THz for each path segment, shape (npoints, N_bands) per segment.
    distances : list of np.ndarray
        Cumulative path distances for each segment (used as the x-axis in band plots),
        shape (npoints,) per segment.
    qpoints : list of np.ndarray
        q-point coordinates for each segment, shape (npoints, 3) per segment.
    """
    atoms, _ = read_crystal_structure(filename=poscar_path)
    phonons = Phonopy(atoms, supercell_matrix)
    force_constants = phonopy_file_IO.read_force_constants_hdf5(filename=fc2_path)
    phonons.force_constants = force_constants
    phonons.symmetrize_force_constants()

    qpoints, connections = get_band_qpoints_and_path_connections(path, npoints=npoints)

    phonons.run_band_structure(qpoints, path_connections=connections, labels=labels)
    bs_dict = phonons.get_band_structure_dict()

    return bs_dict["frequencies"], bs_dict["distances"], qpoints


def run_phonon_mesh(
    poscar_path: str,
    fc2_path: str,
    supercell_matrix: np.ndarray,
    mesh: list,
    is_gamma_center: bool = True,
):
    """Diagonalise the dynamical matrix on a uniform BZ mesh via Phonopy.

    Converts Phonopy output units on return:

    - frequencies: THz → cm⁻¹  (×THzToCm ≈ 33.356)
    - group velocities: THz·Å → m/s  (×THz×Angstrom)

    Parameters
    ----------
    poscar_path : str
        Path to the POSCAR file (primitive cell).
    fc2_path : str
        Path to the second-order force constants HDF5 file.
    supercell_matrix : np.ndarray, shape (3, 3)
        Supercell transformation matrix used when computing the force constants.
    mesh : list of int
        BZ sampling mesh, e.g. ``[20, 20, 20]``.
    is_gamma_center : bool, optional
        Whether the mesh is Γ-centred (default True).

    Returns
    -------
    dict
        Dictionary with keys:

        - ``frequencies_cm``: np.ndarray, shape (N_qpts, N_bands) — phonon frequencies in cm⁻¹.
        - ``weights``: np.ndarray, shape (N_qpts,) — BZ integration weights (integers).
        - ``qpoints``: np.ndarray, shape (N_qpts, 3) — q-point coordinates in reduced coordinates.
        - ``group_velocities_ms``: np.ndarray, shape (N_qpts, N_bands, 3) — Cartesian group
          velocities in m/s.
    """
    atoms, _ = read_crystal_structure(filename=poscar_path)
    phonons = Phonopy(atoms, supercell_matrix)
    force_constants = phonopy_file_IO.read_force_constants_hdf5(filename=fc2_path)
    phonons.force_constants = force_constants
    phonons.symmetrize_force_constants()

    phonons.run_mesh(mesh, is_gamma_center=is_gamma_center, with_group_velocities=True)
    mesh_dict = phonons.get_mesh_dict()

    frequencies_cm = mesh_dict["frequencies"] * THzToCm          # THz → cm⁻¹
    group_velocities_ms = mesh_dict["group_velocities"] * THz * Angstrom  # → m/s

    return {
        "frequencies_cm": frequencies_cm,
        "weights": mesh_dict["weights"],
        "qpoints": mesh_dict["qpoints"],
        "group_velocities_ms": group_velocities_ms,
    }


def save_mesh_data_to_files(
    filename: str,
    frequencies_cm: np.ndarray,
    weights: np.ndarray,
    qpoints: np.ndarray,
    group_velocities_ms: np.ndarray,
):
    """Write mesh data (frequencies, weights, qpoints, group velocities) to HDF5.

    Parameters
    ----------
    filename : str
        Output path without extension; saves to ``{filename}.hdf5``.
    frequencies_cm : np.ndarray, shape (N_qpts, N_bands)
        Phonon frequencies in cm⁻¹.
    weights : np.ndarray, shape (N_qpts,)
        BZ integration weights (integers).
    qpoints : np.ndarray, shape (N_qpts, 3)
        q-point coordinates in reduced coordinates.
    group_velocities_ms : np.ndarray, shape (N_qpts, N_bands, 3)
        Group velocities in m/s.
    """
    compression = "gzip"
    with h5py.File(f"{filename}.hdf5", "w") as w:
        w.create_dataset("frequencies_cm",      data=frequencies_cm,      compression=compression)
        w.create_dataset("weights",             data=weights,             compression=compression)
        w.create_dataset("qpoints",             data=qpoints,             compression=compression)
        w.create_dataset("group_velocities_ms", data=group_velocities_ms, compression=compression)


####
# FUNCTIONS OF FREQUENCY CALCULATIONS
###


def lorentzian_numpy(x, eta):
    """Lorentzian spectral function evaluated with NumPy.

    L(x, η) = (1/π) · η / (x² + η²)

    where η is the **half-width at half-maximum (HWHM)**.

    .. warning::
       This convention differs from the paper (Phys. Rev. X 15, 041041 (2025)),
       where η denotes the full width at half-maximum (FWHM):
       L(x, η) = (1/π) · (η/2) / (x² + (η/2)²).
       The two are consistent when ``eta`` here (disorder linewidth) is passed 
       into the function after dividing by two.

    Parameters
    ----------
    x : np.ndarray
        Frequency offset (ω − ω₀) in cm⁻¹.
    eta : float or np.ndarray
        Lorentzian half-width at half-maximum (HWHM) in cm⁻¹.

    Returns
    -------
    np.ndarray
        Lorentzian values, same shape as ``x``.
    """
    return 1/np.pi * eta / (np.square(x) + np.square(eta))


def flatten_arrays(frequencies, weights, group_velocities):
    """Flatten and sort 2D mesh arrays (N_qpts × N_bands) to 1D, sorted by frequency.

    Scalar speed per mode is the RMS of the 3D velocity:

        \\|v_{q,s}\\| = sqrt((v_x² + v_y² + v_z²) / 3)

    ``weights_sum`` is computed before repeating weights to preserve the correct
    normalisation denominator for the VDOS integral.

    Parameters
    ----------
    frequencies : np.ndarray, shape (N_qpts, N_bands)
        Phonon frequencies in cm⁻¹.
    weights : np.ndarray, shape (N_qpts,)
        BZ integration weights (integers).
    group_velocities : np.ndarray, shape (N_qpts, N_bands, 3)
        Group velocities in m/s.

    Returns
    -------
    frequencies_flat : np.ndarray, shape (N_qpts × N_bands,)
        Sorted flattened frequencies in cm⁻¹.
    weights_flat : np.ndarray, shape (N_qpts × N_bands,)
        Weights repeated N_bands times and sorted to match ``frequencies_flat``.
    speed_flat : np.ndarray, shape (N_qpts × N_bands,)
        RMS group speeds in m/s, sorted to match ``frequencies_flat``.
    weights_sum : int
        Sum of the original (un-repeated) BZ weights; used as the VDOS normalisation denominator.
    """
    N_qpts, N_bands = frequencies.shape
    frequencies_flat = frequencies.flatten()

    weights_flat = np.repeat(weights, N_bands)
    weights_sum = weights.sum()  # normalisation: sum before repeating

    speed_2d = np.sqrt(
        np.square(group_velocities).sum(axis=2) / 3
    )

    speed_flat = speed_2d.flatten()

    sort_idx = np.argsort(frequencies_flat)
    frequencies_flat = frequencies_flat[sort_idx]
    weights_flat = weights_flat[sort_idx]
    speed_flat = speed_flat[sort_idx]

    return frequencies_flat, weights_flat, speed_flat, weights_sum


def calculate_vdos_and_average_speed_with_frequency(
    frequencies_flat: np.ndarray,
    weights_flat: np.ndarray,
    speed_flat: np.ndarray,
    gamma_min_plateau: float,
    structure_file: str,
    weights_sum: int):
    """Compute the VDOS and frequency-resolved mean group speed via Lorentzian broadening.

    The vibrational density of states and frequency-resolved group speed are:

        g(ω) = [1/(V·Σ_q w_q)] · Σ_{q,n} w_q · L(ω−ω_{q,n}, η)

        v_g(ω) = [Σ_{q,n} w_q·\\|v_{q,n}\\|·L(ω−ω_{q,n}, η)] / [Σ_{q,n} w_q·L(ω−ω_{q,n}, η)]

    where η = ``gamma_min_plateau`` [cm⁻¹] is the Lorentzian half-width at half-maximum.
    ``v_g`` is set to 0 where VDOS ≈ 0 to avoid division by zero.

    Output VDOS is converted from cm·Å⁻³ to THz⁻¹·nm⁻³:

        conversion_factor = 1000 · THzToCm / (2π)

    Parameters
    ----------
    frequencies_flat : np.ndarray, shape (N,)
        Sorted flattened mode frequencies in cm⁻¹ (output of :func:`flatten_arrays`).
    weights_flat : np.ndarray, shape (N,)
        Sorted flattened BZ weights (output of :func:`flatten_arrays`).
    speed_flat : np.ndarray, shape (N,)
        Sorted flattened RMS group speeds in m/s (output of :func:`flatten_arrays`).
    gamma_min_plateau : float
        Lorentzian half-width (HWHM) η in cm⁻¹. Controls spectral resolution.
    structure_file : str
        Path to the structure file (POSCAR/CIF/XYZ); used to read the unit-cell volume.
    weights_sum : int
        Sum of un-repeated BZ weights (output of :func:`flatten_arrays`).

    Returns
    -------
    vdos_return : np.ndarray
        VDOS in THz⁻¹ nm⁻³, evaluated on ``frequencies_bin``.
    speed_average_return : np.ndarray
        Frequency-resolved mean group speed in m/s, evaluated on ``frequencies_bin``.
    frequencies_bin : np.ndarray
        Frequency grid in cm⁻¹ on which VDOS and speed are evaluated.
    """
    width_frequency = gamma_min_plateau / 5
    min_frequency = frequencies_flat[3]
    max_frequency = np.amax(frequencies_flat) + 50

    frequencies_bin = np.arange(min_frequency - 5*gamma_min_plateau, max_frequency + 5*gamma_min_plateau, width_frequency)
    speed_average_bin = np.zeros(len(frequencies_bin))
    vdos_bin = np.zeros(len(frequencies_bin))

    for idx_bin in tqdm(range(len(frequencies_bin)), desc="VDOS & v(w)"):
        frequency = frequencies_bin[idx_bin]
        lorentzian_distr = lorentzian_numpy(frequencies_flat - frequency, eta=gamma_min_plateau)
        vdos_bin[idx_bin] = (weights_flat * lorentzian_distr).sum()
        speed_average_bin[idx_bin] = (weights_flat * speed_flat * lorentzian_distr).sum()

    cell = read(structure_file)
    volume_A3 = cell.get_volume()

    speed_average_return = np.where(
        np.isclose(vdos_bin, 0.0),
        0.0,
        speed_average_bin / vdos_bin,
    )

    vdos_return = vdos_bin / (volume_A3 * weights_sum)

    # conversion from cm·Å⁻³ to THz⁻¹·nm⁻³ (includes 1/(2π) for angular frequency)
    conversion_factor = 1000 * THzToCm / (2*np.pi)
    vdos_return *= conversion_factor

    return vdos_return, speed_average_return, frequencies_bin


def save_vdos_speed_data_to_files(
    filename: str,
    frequencies_bin: np.ndarray,
    vdos_return: np.ndarray,
    speed_average_return: np.ndarray,
):
    """Write VDOS and mean group speed vs frequency to HDF5.

    Parameters
    ----------
    filename : str
        Output path without extension; saves to ``{filename}.hdf5``.
    frequencies_bin : np.ndarray
        Frequency grid in cm⁻¹.
    vdos_return : np.ndarray
        VDOS in THz⁻¹ nm⁻³.
    speed_average_return : np.ndarray
        Frequency-resolved mean group speed in m/s.
    """
    compression = "gzip"
    with h5py.File(f"{filename}.hdf5", "w") as w:
        w.create_dataset("frequencies_bin", data=frequencies_bin, compression=compression)
        w.create_dataset("vdos_return", data=vdos_return, compression=compression)
        w.create_dataset("speed_average_return", data=speed_average_return, compression=compression)


def flatten_arrays_freq_only(frequencies, weights):
    """Flatten and sort 2D mesh frequency array to 1D (no group velocities).

    Use this variant when only the VDOS is needed and group speed is not required,
    e.g. for disordered structures where group velocity is ill-defined.

    Parameters
    ----------
    frequencies : np.ndarray, shape (N_qpts, N_bands)
        Phonon frequencies in cm⁻¹.
    weights : np.ndarray, shape (N_qpts,)
        BZ integration weights (integers).

    Returns
    -------
    frequencies_flat : np.ndarray, shape (N_qpts × N_bands,)
        Sorted flattened frequencies in cm⁻¹.
    weights_flat : np.ndarray, shape (N_qpts × N_bands,)
        Weights repeated N_bands times and sorted to match ``frequencies_flat``.
    weights_sum : int
        Sum of the original (un-repeated) BZ weights.
    """
    N_qpts, N_bands = frequencies.shape
    frequencies_flat = frequencies.flatten()

    weights_flat = np.repeat(weights, N_bands)
    weights_sum = weights.sum()  # normalisation: sum before repeating

    sort_idx = np.argsort(frequencies_flat)
    frequencies_flat = frequencies_flat[sort_idx]
    weights_flat = weights_flat[sort_idx]

    return frequencies_flat, weights_flat, weights_sum


def calculate_vdos_with_frequency(
    frequencies_flat: np.ndarray,
    weights_flat: np.ndarray,
    gamma_min_plateau: float,
    structure_file: str,
    weights_sum: int):
    """Compute the VDOS via Lorentzian broadening (no group speed).

    Same Lorentzian broadening as
    :func:`calculate_vdos_and_average_speed_with_frequency` but omits the
    weighted speed accumulation. Use for disordered structures where group
    velocity is ill-defined and only the VDOS is needed.

    Parameters
    ----------
    frequencies_flat : np.ndarray, shape (N,)
        Sorted flattened mode frequencies in cm⁻¹
        (output of :func:`flatten_arrays_freq_only`).
    weights_flat : np.ndarray, shape (N,)
        Sorted flattened BZ weights (output of :func:`flatten_arrays_freq_only`).
    gamma_min_plateau : float
        Lorentzian half-width (HWHM) η in cm⁻¹. Controls spectral resolution.
    structure_file : str
        Path to the structure file (POSCAR/CIF/XYZ); used to read the unit-cell volume.
    weights_sum : int
        Sum of un-repeated BZ weights (output of :func:`flatten_arrays_freq_only`).

    Returns
    -------
    vdos_return : np.ndarray
        VDOS in THz⁻¹ nm⁻³, evaluated on ``frequencies_bin``.
    frequencies_bin : np.ndarray
        Frequency grid in cm⁻¹ on which the VDOS is evaluated.
    """
    width_frequency = gamma_min_plateau / 5
    min_frequency = frequencies_flat[3]
    max_frequency = np.amax(frequencies_flat) + 50

    frequencies_bin = np.arange(min_frequency - 5*gamma_min_plateau, max_frequency + 5*gamma_min_plateau, width_frequency)
    vdos_bin = np.zeros(len(frequencies_bin))

    for idx_bin in tqdm(range(len(frequencies_bin)), desc="VDOS & v(w)"):
        frequency = frequencies_bin[idx_bin]
        lorentzian_distr = lorentzian_numpy(frequencies_flat - frequency, eta=gamma_min_plateau)
        vdos_bin[idx_bin] = (weights_flat * lorentzian_distr).sum()

    cell = read(structure_file)
    volume_A3 = cell.get_volume()

    vdos_return = vdos_bin / (volume_A3 * weights_sum)

    # conversion from cm·Å⁻³ to THz⁻¹·nm⁻³ (includes 1/(2π) for angular frequency)
    conversion_factor = 1000 * THzToCm / (2*np.pi)
    vdos_return *= conversion_factor

    return vdos_return, frequencies_bin


def save_vdos_data_to_files(
    filename: str,
    frequencies_bin: np.ndarray,
    vdos_return: np.ndarray,
):
    """Write VDOS vs frequency to HDF5.

    Parameters
    ----------
    filename : str
        Output path without extension; saves to ``{filename}.hdf5``.
    frequencies_bin : np.ndarray
        Frequency grid in cm⁻¹.
    vdos_return : np.ndarray
        VDOS in THz⁻¹ nm⁻³.
    """
    compression = "gzip"
    with h5py.File(f"{filename}.hdf5", "w") as w:
        w.create_dataset("frequencies_bin", data=frequencies_bin, compression=compression)
        w.create_dataset("vdos_return", data=vdos_return, compression=compression)


###
# PREPARE FILES FOR FITTING + INFLUENCE OF DISORDER LINEWIDTH ON VDOS
###


def prepare_fitting_inputs(crystal_poscar, disordered_poscar,
                           disordered_vdos_save, shifted_save,
                           n_interp=2680):
    """Load and prepare all inputs needed for disorder-linewidth fitting.

    Reads mass densities from POSCAR files and VDOS/speed data from the HDF5 files
    written by :func:`save_vdos_data_to_files` and
    :func:`save_vdos_speed_data_to_files`.

    The density-shifted crystal VDOS and speed are resampled onto a uniform grid of
    ``n_interp`` points to reduce the size of the (N_dis × N_interp)
    frequency-difference matrix built inside
    :func:`evaluate_linewidth_and_model_prediction`.

    Parameters
    ----------
    crystal_poscar : str
        Path to the crystal POSCAR file; used to compute crystal mass density.
    disordered_poscar : str
        Path to the disordered structure POSCAR file; used to compute disordered
        mass density.
    disordered_vdos_save : str
        Path (without ``.hdf5``) to the disordered VDOS HDF5 file produced by
        :func:`save_vdos_data_to_files`.
    shifted_save : str
        Path (without ``.hdf5``) to the density-shifted crystal VDOS + speed HDF5
        file produced by :func:`save_vdos_speed_data_to_files` after the
        frequency-shift correction.
    n_interp : int, optional
        Number of interpolation points for the crystal frequency grid (default 2680).

    Returns
    -------
    density_crystal : float
        Mass density of the crystal in g/cm³.
    density_disordered : float
        Mass density of the disordered structure in g/cm³.
    freq_disordered : np.ndarray
        Frequency grid for the disordered VDOS in cm⁻¹.
    vdos_disordered : np.ndarray
        Disordered VDOS in THz⁻¹ nm⁻³.
    interp_shifted_freq_crystal : np.ndarray
        Uniform frequency grid for the interpolated crystal data in cm⁻¹.
    interp_shifted_vdos_crystal : np.ndarray
        Crystal VDOS (density-shifted) interpolated onto
        ``interp_shifted_freq_crystal`` in THz⁻¹ nm⁻³.
    interp_shifted_speed_crystal : np.ndarray
        Mean group speed (density-shifted) interpolated onto
        ``interp_shifted_freq_crystal`` in m/s.
    """
    density_crystal = obtain_density(read(crystal_poscar))
    density_disordered = obtain_density(read(disordered_poscar))

    with h5py.File(f"{disordered_vdos_save}.hdf5", "r") as f:
        freq_disordered = np.asarray(f["frequencies_bin"])   # [cm⁻¹]
        vdos_disordered = np.asarray(f["vdos_return"])       # [THz⁻¹ nm⁻³]

    with h5py.File(f"{shifted_save}.hdf5", "r") as f:
        shifted_freq_crystal = np.asarray(f["frequencies_bin"])          # [cm⁻¹]
        shifted_vdos_crystal = np.asarray(f["vdos_return"])              # [THz⁻¹ nm⁻³]
        shifted_speed_crystal = np.asarray(f["speed_average_return"])    # [m/s]

    # sparsify to reduce frequency_bin_differences matrix size in fitting
    interp_shifted_freq_crystal = np.linspace(shifted_freq_crystal[0],
                                              shifted_freq_crystal[-1],
                                              n_interp)
    interp_shifted_vdos_crystal = interp1d(shifted_freq_crystal,
                                           shifted_vdos_crystal)(interp_shifted_freq_crystal)
    interp_shifted_speed_crystal = interp1d(shifted_freq_crystal,
                                            shifted_speed_crystal)(interp_shifted_freq_crystal)

    return (density_crystal,
            density_disordered,
            freq_disordered,
            vdos_disordered,
            interp_shifted_freq_crystal,
            interp_shifted_vdos_crystal,
            interp_shifted_speed_crystal)


def evaluate_linewidth_and_model_prediction(
    density_crystal,
    density_disordered,
    freq_disordered,
    vdos_disordered,
    interp_shifted_freq_crystal,
    interp_shifted_vdos_crystal,
    interp_shifted_speed_crystal,
    L, R,
):
    """Evaluate the disorder linewidth model and predicted PDC VDOS for given L and R.

    Two scattering mechanisms contribute (see Phys. Rev. X 15, 041041 (2025)):

        Γ_defect(ω)  = R · ω² · g^{shifted}(ω) · (ρ_dis/ρ_crys)   [cm⁻¹]
        Γ_Casimir(ω) = v_g(ω) / L · 1e-2 · THzToCm            [cm⁻¹]

    where L is in Å and the unit path is m/s ÷ Å = 1e10 s⁻¹ = 1e-2 THz → ×THzToCm → cm⁻¹.

    The predicted disordered VDOS (PDC model) is the crystal VDOS convolved with a
    Lorentzian of half-width Γ(ω)/2:

        g_model(ω) = ρ_dis · Σ_{ω'} [g_crystal(ω')/ρ_crys] · L(ω−ω', Γ(ω')/2) · Δω'

    implemented as a matrix–vector broadcast over the (N_dis × N_interp)
    frequency-difference matrix.

    Parameters
    ----------
    density_crystal : float
        Crystal mass density in g/cm³.
    density_disordered : float
        Disordered structure mass density in g/cm³.
    freq_disordered : np.ndarray, shape (N_dis,)
        Frequency grid for the disordered VDOS in cm⁻¹.
    vdos_disordered : np.ndarray, shape (N_dis,)
        Measured disordered VDOS in THz⁻¹ nm⁻³. Not consumed internally;
        passed for caller convenience so all outputs of :func:`prepare_fitting_inputs`
        can be forwarded directly.
    interp_shifted_freq_crystal : np.ndarray, shape (N_interp,)
        Uniform crystal frequency grid in cm⁻¹.
    interp_shifted_vdos_crystal : np.ndarray, shape (N_interp,)
        Density-shifted crystal VDOS in THz⁻¹ nm⁻³.
    interp_shifted_speed_crystal : np.ndarray, shape (N_interp,)
        Density-shifted mean group speed in m/s.
    L : float
        Grain-boundary mean free path (Casimir length) in Å.
    R : float
        Defect scattering coefficient in cm THz nm³.

    Returns
    -------
    vdos_PDC : np.ndarray, shape (N_dis,)
        PDC model VDOS in THz⁻¹ nm⁻³, evaluated on ``freq_disordered``.
    disorder_linewidth : np.ndarray, shape (N_interp,)
        Total disorder linewidth Γ(ω) = Γ_defect + Γ_Casimir in cm⁻¹.
    defect_linewidth : np.ndarray, shape (N_interp,)
        Defect (Rayleigh) scattering contribution Γ_defect(ω) in cm⁻¹.
    Casimir_model_linewidth : np.ndarray, shape (N_interp,)
        Casimir (grain-boundary) scattering contribution Γ_Casimir(ω) in cm⁻¹.
    """
    X = interp_shifted_vdos_crystal / density_crystal
    width_frequency_input = interp_shifted_freq_crystal[1] - interp_shifted_freq_crystal[0]
    frequency_bin_differences = freq_disordered.reshape(-1, 1) - interp_shifted_freq_crystal.reshape(1, -1)

    defect_linewidth = R * np.square(interp_shifted_freq_crystal) * interp_shifted_vdos_crystal * density_disordered / density_crystal

    # v [m/s] / L [Å] * 1e-2 (Å⁻¹→THz) * THzToCm → cm⁻¹
    Casimir_model_linewidth = interp_shifted_speed_crystal / L * 1e-2 * THzToCm

    disorder_linewidth = defect_linewidth + Casimir_model_linewidth

    vdos_PDC = (lorentzian_numpy(frequency_bin_differences, eta=disorder_linewidth.reshape(1, -1)/2) * X.reshape(1, -1)).sum(axis=1) * width_frequency_input
    vdos_PDC *= density_disordered

    return vdos_PDC, disorder_linewidth, defect_linewidth, Casimir_model_linewidth


#####################
# PARAMETER FITTING #
#####################

def lorentzian_torch(x, eta):
    """Lorentzian spectral function evaluated with PyTorch (supports autodiff).

    L(x, η) = (1/π) · η / (x² + η²)

    where η is the half-width at half-maximum (HWHM). Used inside :class:`PDCModel`
    so that gradients flow through the Lorentzian during L-BFGS fitting.

    Parameters
    ----------
    x : torch.Tensor
        Frequency offset (ω − ω₀) in cm⁻¹.
    eta : torch.Tensor
        Lorentzian half-width at half-maximum (HWHM) in cm⁻¹.

    Returns
    -------
    torch.Tensor
        Lorentzian values, same shape as ``x``.
    """
    return 1/torch.pi * eta / (torch.square(x) + torch.square(eta))


class PDCModel(torch.nn.Module):
    """PyTorch model for fitting the grain-boundary length L and defect scattering
    coefficient R to the measured disordered VDOS.

    The forward pass computes the PDC (phonon disorder convolution) model VDOS:
      1. Γ_defect = R·1e-6 · ω² · g^{DR} · (ρ_dis/ρ_crys)
      2. Γ_Casimir = v_g / (L·1e1 [Å]) · 1e-2 · THzToCm
      3. Γ = Γ_defect + Γ_Casimir
      4. g_model = Σ_{ω'} X(ω') · L(ω−ω', Γ(ω')/2) · Δω', here L(x, η), where η is the half width

    where X = g_crystal/ρ_crys is passed in as an argument (kept in notebook scope
    so the LBFGS closure can reference it directly).

    Learnable parameter vector: model_params = [L, R] with L in nm and R in units of 1e-6.
    Call model_params.detach().cpu().numpy() after fitting to read the optimised values.
    """

    def __init__(
        self,
        L0: float,
        R0: float,
        density_crystal: float,
        density_disordered: float,
        freq_disordered: np.ndarray,
        interp_shifted_freq_crystal: np.ndarray,
        interp_shifted_vdos_crystal: np.ndarray,
        interp_shifted_speed_crystal: np.ndarray,
    ):
        super(PDCModel, self).__init__()

        self.density_crystal = density_crystal
        self.density_disordered = density_disordered

        self.interp_shifted_freq_crystal = torch.from_numpy(interp_shifted_freq_crystal)
        self.interp_shifted_vdos_crystal = torch.from_numpy(interp_shifted_vdos_crystal)
        self.interp_shifted_speed_crystal = torch.from_numpy(interp_shifted_speed_crystal)

        # frequency difference matrix (N_ref × N_sparse) — computed once, stored as buffer
        self.frequency_differences = torch.from_numpy(
            freq_disordered.reshape(-1, 1) - interp_shifted_freq_crystal.reshape(1, -1)
        )
        self.width_frequency = float(interp_shifted_freq_crystal[1] - interp_shifted_freq_crystal[0])

        self.model_params = torch.nn.Parameter(
            torch.from_numpy(np.array([L0, R0])), requires_grad=True
        )
        self.double()

    def forward(self, X):
        output = torch.zeros(len(self.frequency_differences), dtype=torch.double)

        # Γ_defect = R·1e-6 · ω² · g^{shifted} · (ρ_dis/ρ_crys)
        defect_linewidth = (
            self.model_params[1] * 1e-6
            * torch.square(self.interp_shifted_freq_crystal)
            * self.interp_shifted_vdos_crystal
            * self.density_disordered / self.density_crystal
        )

        # Γ_Casimir = v / (L [nm] · 10 [Å/nm]) · 1e-2 · THzToCm  → cm⁻¹
        Casimir_model_linewidth = (
            self.interp_shifted_speed_crystal / self.model_params[0]
            * 1e-1 * 1e-2 * THzToCm
        )

        disorder_linewidth = defect_linewidth + Casimir_model_linewidth

        output = (
            lorentzian_torch(self.frequency_differences, eta=disorder_linewidth.reshape(1, -1) / 2.0)
            * X.reshape(1, -1)
        ).sum(axis=1) * self.width_frequency

        return output