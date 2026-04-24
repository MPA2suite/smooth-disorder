"""
test_disorder_linewidth.py — Regression tests for disorder linewidth workflow outputs
======================================================================================

These tests verify that the HDF5 output files produced by the disorder linewidth
workflow (workflows/7a_workflow_precompute_quantities.py and tutorials in tutorials/disorder_linewidth)
match the reference files committed to the repository in tutorials/disorder_linewidth/dl_test/.

Tests skip gracefully when the target directories (dl_workflow, dl_tutorial) do not
exist, since those directories are gitignored and require running the workflow first.
"""

import os
import numpy as np
import pytest
import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_VDOS_SPEED_FILE = "crystal_vdos_group_vel.hdf5"
_VDOS_SPEED_KEYS = ["frequencies_bin", "vdos_return", "speed_average_return"]

_DISORDERED_FILE = "disordered_vdos.hdf5"
_DISORDERED_KEYS = ["frequencies_bin", "vdos_return"]

_REDUCED_FILE = "reduced_density_crystal_vdos_group_vel.hdf5"
_REDUCED_KEYS = ["frequencies_bin", "vdos_return", "speed_average_return"]

_MODEL_PARAMS_FILE = "model_parameters.hdf5"
_MODEL_PARAMS_KEYS = ["final_loss", "final_model_params"]


def _assert_hdf5_arrays_match(ref_path, candidate_path, keys):
    """Compare each dataset between two HDF5 files within rtol=1e-10."""
    with h5py.File(ref_path, "r") as ref_f, h5py.File(candidate_path, "r") as cand_f:
        for key in keys:
            np.testing.assert_allclose(
                cand_f[key][:], ref_f[key][:], rtol=1e-10,
                err_msg=f"Mismatch in '{key}' between {candidate_path} and {ref_path}",
            )


# ---------------------------------------------------------------------------
# dl_workflow vs dl_test
# ---------------------------------------------------------------------------

@pytest.mark.disorder_linewidth
class TestDisorderLinewidthWorkflowOutputs:
    """Compare dl_workflow output files against dl_test reference files."""

    @pytest.fixture(scope="class")
    def dl_workflow_dir(self):
        path = os.path.join(REPO_ROOT, "tutorials", "disorder_linewidth", "dl_workflow")
        if not os.path.isdir(path):
            pytest.skip(
                "dl_workflow directory not found — run "
                "tutorials/disorder_linewidth/7a_workflow_precompute_quantities.py, "
                "7b_fit_dl_parameters.py, and 7c_diffusivity_decomposition.py first"
            )
        return path

    def test_crystal_vdos_group_vel(self, dl_test_dir, dl_workflow_dir):
        """crystal_vdos_group_vel.hdf5: frequencies, VDOS, and mean group speed."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _VDOS_SPEED_FILE),
            os.path.join(dl_workflow_dir, _VDOS_SPEED_FILE),
            _VDOS_SPEED_KEYS,
        )

    def test_disordered_vdos(self, dl_test_dir, dl_workflow_dir):
        """disordered_vdos.hdf5: frequencies and VDOS for the disordered system."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _DISORDERED_FILE),
            os.path.join(dl_workflow_dir, _DISORDERED_FILE),
            _DISORDERED_KEYS,
        )

    def test_reduced_density_crystal_vdos_group_vel(self, dl_test_dir, dl_workflow_dir):
        """reduced_density_crystal_vdos_group_vel.hdf5: density-shifted crystal VDOS."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _REDUCED_FILE),
            os.path.join(dl_workflow_dir, _REDUCED_FILE),
            _REDUCED_KEYS,
        )


# ---------------------------------------------------------------------------
# dl_tutorial vs dl_test
# ---------------------------------------------------------------------------

@pytest.mark.disorder_linewidth
class TestDisorderLinewidthTutorialOutputs:
    """Compare dl_tutorial output files against dl_test reference files."""

    @pytest.fixture(scope="class")
    def dl_tutorial_dir(self):
        path = os.path.join(REPO_ROOT, "tutorials", "disorder_linewidth", "dl_tutorial")
        if not os.path.isdir(path):
            pytest.skip(
                "dl_tutorial directory not found — run the notebook series "
                "in tutorials/disorder_linewidth/ first"
            )
        return path

    def test_crystal_vdos_group_vel(self, dl_test_dir, dl_tutorial_dir):
        """crystal_vdos_group_vel.hdf5: frequencies, VDOS, and mean group speed."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _VDOS_SPEED_FILE),
            os.path.join(dl_tutorial_dir, _VDOS_SPEED_FILE),
            _VDOS_SPEED_KEYS,
        )

    def test_disordered_vdos(self, dl_test_dir, dl_tutorial_dir):
        """disordered_vdos.hdf5: frequencies and VDOS for the disordered system."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _DISORDERED_FILE),
            os.path.join(dl_tutorial_dir, _DISORDERED_FILE),
            _DISORDERED_KEYS,
        )

    def test_reduced_density_crystal_vdos_group_vel(self, dl_test_dir, dl_tutorial_dir):
        """reduced_density_crystal_vdos_group_vel.hdf5: density-shifted crystal VDOS."""
        _assert_hdf5_arrays_match(
            os.path.join(dl_test_dir, _REDUCED_FILE),
            os.path.join(dl_tutorial_dir, _REDUCED_FILE),
            _REDUCED_KEYS,
        )


# ---------------------------------------------------------------------------
# workflows/dl_workflow vs workflows/dl_test
# ---------------------------------------------------------------------------

@pytest.mark.dl_workflow
class TestDisorderLinewidthWorkflowsOutputs:
    """Compare workflows/dl_workflow HDF5 files against workflows/dl_test references."""

    @pytest.fixture(scope="class")
    def dl_workflow_scripts_dir(self):
        path = os.path.join(REPO_ROOT, "workflows", "dl_workflow")
        if not os.path.isdir(path):
            pytest.skip(
                "workflows/dl_workflow directory not found — run "
                "workflows/2a_DL_workflow_precompute.py first"
            )
        return path

    def test_crystal_vdos_group_vel(self, workflows_dl_test_dir, dl_workflow_scripts_dir):
        """crystal_vdos_group_vel.hdf5: frequencies, VDOS, and mean group speed."""
        _assert_hdf5_arrays_match(
            os.path.join(workflows_dl_test_dir, _VDOS_SPEED_FILE),
            os.path.join(dl_workflow_scripts_dir, _VDOS_SPEED_FILE),
            _VDOS_SPEED_KEYS,
        )

    def test_disordered_vdos(self, workflows_dl_test_dir, dl_workflow_scripts_dir):
        """disordered_vdos.hdf5: frequencies and VDOS for the disordered system."""
        _assert_hdf5_arrays_match(
            os.path.join(workflows_dl_test_dir, _DISORDERED_FILE),
            os.path.join(dl_workflow_scripts_dir, _DISORDERED_FILE),
            _DISORDERED_KEYS,
        )

    def test_reduced_density_crystal_vdos_group_vel(self, workflows_dl_test_dir, dl_workflow_scripts_dir):
        """reduced_density_crystal_vdos_group_vel.hdf5: density-shifted crystal VDOS."""
        _assert_hdf5_arrays_match(
            os.path.join(workflows_dl_test_dir, _REDUCED_FILE),
            os.path.join(dl_workflow_scripts_dir, _REDUCED_FILE),
            _REDUCED_KEYS,
        )

    def test_model_parameters(self, workflows_dl_test_dir, dl_workflow_scripts_dir):
        """model_parameters.hdf5: fitted L/R parameters and final loss."""
        _assert_hdf5_arrays_match(
            os.path.join(workflows_dl_test_dir, _MODEL_PARAMS_FILE),
            os.path.join(dl_workflow_scripts_dir, _MODEL_PARAMS_FILE),
            _MODEL_PARAMS_KEYS,
        )