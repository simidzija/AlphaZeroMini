import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

import torch

def test_action_dist():
    num_out_channels = 4
    board_size = 5

    logits = torch.full((1, num_out_channels, board_size, board_size), -torch.inf)
    logits[0, 1, 2, 3] = 1
    logits[0, 2, 1, 4] = 2
    non_zero_entries = 2

    actions, probs = utils.action_dist(logits)

    assert actions.shape == torch.Size([non_zero_entries, 1, 3]), f'got actions.shape = {actions.shape}'
    assert len(probs) == non_zero_entries