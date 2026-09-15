import numpy as np
import pytest

from qm_atlas.workflows.utils import filters

TEST_VALUES = [("conformers_properties.txt", [26])]


@pytest.mark.parametrize("filename, target_idxs", TEST_VALUES)
def test_group_dbscan(filename, target_idxs):

    # Load descriptors for molecules
    filename = "tests/resources/" + filename
    values = np.loadtxt(filename)

    # Get groups of indicies and list of outlier indicies
    idxs_groups, outliers = filters.group_dbscan(values)

    # Select first from each group and all outliers
    filtered_indicies = [idxs[0] for idxs in idxs_groups]
    filtered_indicies += list(outliers)

    # Assure that the target indicies are found for the filter
    assert set(target_idxs).issubset(set(filtered_indicies))

    return
