import chess
import torch

def test_action_mask():
    # test 1
    features = torch.zeros(4,5,5, dtype=torch.bool)
    features[0, 2, 3] = 1 # white at (2,3)
    features[1, 4, 4] = 1 # black at (4,4)
    features[2, ...] = 0 # white to play
    features[3, ...] = 7 # 7 moves played

    mask_idxs = chess.action_mask(features).nonzero()
    expected_idxs = torch.tensor([[0, 2, 3],
                                  [1, 2, 3],
                                  [2, 2, 3],
                                  [3, 2, 3]])

    assert torch.all(mask_idxs == expected_idxs)

    # test 2
    features = torch.zeros(4,5,5, dtype=torch.bool)
    features[0, 2, 3] = 1 # white at (2,3)
    features[1, 4, 4] = 1 # black at (4,4)
    features[2, ...] = 1 # black to play
    features[3, ...] = 7 # 7 moves played

    mask_idxs = chess.action_mask(features).nonzero()
    expected_idxs = torch.tensor([[0, 4, 4],
                                  [2, 4, 4]])

    assert torch.all(mask_idxs == expected_idxs)

    
