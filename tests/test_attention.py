import pytest
torch = pytest.importorskip('torch')
from cometa.attention import tree_visibility, tree_position_ids, tree_readout_positions, tree_attention_mask


def test_toy_tree_visibility_and_reset_positions():
    visible = tree_visibility(2, [2, 1])
    assert visible.tolist() == [
        [1, 0, 0, 0, 0], [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 0, 0, 1]]
    assert tree_position_ids(2, [2, 1]).tolist() == [[0, 1, 2, 3, 2]]
    assert tree_readout_positions(2, [2, 1]) == [3, 4]
    mask = tree_attention_mask(2, [2, 1])
    assert mask.shape == (1, 1, 5, 5)
    assert mask[0, 0, 4, 2] < -1e20 and mask[0, 0, 4, 1] == 0


@pytest.mark.parametrize('prefix,branches', [(0, [1]), (1, []), (1, [0])])
def test_reject_invalid_trees(prefix, branches):
    with pytest.raises(ValueError):
        tree_visibility(prefix, branches)
