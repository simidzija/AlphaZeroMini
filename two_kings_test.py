import two_kings
import torch

def test_action_mask():
    # test 1
    state = torch.zeros(1,3,5,5, dtype=torch.bool)
    state[0, 0, 2, 3] = 1 # P1 at (2,3)
    state[0, 1, 4, 4] = 1 # P2 at (4,4)
    state[0, 2, ...] = 7 # 7 moves played

    mask_idxs = two_kings.action_mask(state)[0].nonzero()
    expected_idxs = torch.tensor([[0, 2, 3],
                                  [1, 2, 3],
                                  [2, 2, 3],
                                  [3, 2, 3]])

    assert torch.all(mask_idxs == expected_idxs)

    # test 2
    state = torch.zeros(1,3,5,5, dtype=torch.bool)
    state[0, 0, 4, 4] = 1 # P1 at (4,4)
    state[0, 1, 2, 3] = 1 # P2 at (2,3)
    state[0, 2, ...] = 7 # 7 moves played

    mask_idxs = two_kings.action_mask(state)[0].nonzero()
    expected_idxs = torch.tensor([[0, 4, 4],
                                  [2, 4, 4]])

    assert torch.all(mask_idxs == expected_idxs)