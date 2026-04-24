"""
test_bne_workflow.py — Regression tests for BNE workflow outputs
================================================================

Compares entropy HDF5 files produced by workflows/1a_BNE_workflow.py
against the reference files committed in workflows/bne_test/.

Tests are skipped when workflows/bne_workflow/ is absent.
"""

import os
import numpy as np
import pytest
import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BNE_LAE_SIZES = list(range(10, 31))
_BNE_KEYS = ["probabilities", "entropy", "number_of_atoms"]


def _assert_hdf5_arrays_match(ref_path, candidate_path, keys):
    """Compare each dataset between two HDF5 files within rtol=1e-10."""
    with h5py.File(ref_path, "r") as ref_f, h5py.File(candidate_path, "r") as cand_f:
        for key in keys:
            np.testing.assert_allclose(
                cand_f[key][:], ref_f[key][:], rtol=1e-10,
                err_msg=f"Mismatch in '{key}' between {candidate_path} and {ref_path}",
            )


@pytest.mark.bne_workflow
class TestBneWorkflowOutputs:
    """Compare workflows/bne_workflow entropy files against workflows/bne_test references."""

    @pytest.fixture(scope="class")
    def bne_workflow_dir(self):
        path = os.path.join(
            REPO_ROOT, "workflows", "bne_workflow", "bond_network_entropy", "structure_0"
        )
        if not os.path.isdir(path):
            pytest.skip(
                "workflows/bne_workflow/bond_network_entropy/structure_0 not found — "
                "run workflows/1a_BNE_workflow.py first"
            )
        return path

    @pytest.mark.parametrize("n", _BNE_LAE_SIZES)
    def test_entropy_number(self, workflows_bne_test_dir, bne_workflow_dir, n):
        """entropy_number_{n}.hdf5: probabilities, entropy scalar, and number_of_atoms."""
        filename = f"entropy_number_{n}.hdf5"
        _assert_hdf5_arrays_match(
            os.path.join(workflows_bne_test_dir, filename),
            os.path.join(bne_workflow_dir, filename),
            _BNE_KEYS,
        )
