import train
from network import Network
from two_kings import EnvTwoKings, action_mask

import torch

net = Network(
    num_in_channels=4,
    board_size=5,
    num_filters=8,
    kernel_size=3,
    num_res_blocks=6,
    num_policy_filters=2,
    num_out_channels=4,
    value_hidden_layer_size=32,
    action_mask=action_mask
)

def test_self_play_game():
    env = EnvTwoKings()
    n_simulations = 10
    c_putc = 0.1
    temp = 1

    buffer = train.self_play_game(env, net, n_simulations=n_simulations,
                                  c_putc=c_putc, temp=temp)

    assert 4 <= len(buffer) <= 20

test_self_play_game()