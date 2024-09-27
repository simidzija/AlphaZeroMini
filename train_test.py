from train import train, self_play_game
from network import Network
from two_kings import EnvTwoKings, action_mask

import torch

num_in_channels=4
board_size=5
num_filters=8
kernel_size=3
num_res_blocks=6
num_policy_filters=2
num_out_channels=4
value_hidden_layer_size=32

net = Network(
    num_in_channels=num_in_channels,
    board_size=board_size,
    num_filters=num_filters,
    kernel_size=kernel_size,
    num_res_blocks=num_res_blocks,
    num_policy_filters=num_policy_filters,
    num_out_channels=num_out_channels,
    value_hidden_layer_size=value_hidden_layer_size,
    action_mask=action_mask
)

def test_self_play_game():
    env = EnvTwoKings()
    n_simulations = 10
    c_puct = 0.1
    temp = 1

    buffer = self_play_game(env, 
                            net, 
                            n_simulations=n_simulations,
                            c_puct=c_puct,
                            alpha_dir=1.0,
                            eps_dir=0.5,
                            temp=temp)

    assert 4 <= len(buffer) <= 20
    state, action, value = buffer[0]
    assert state.shape == torch.Size([1, num_in_channels, board_size, board_size]), f'got shape {state.shape}'
    assert action.shape == torch.Size([1, 3]), f'got shape {action.shape}'
    assert value.shape == torch.Size([1, 1]), f'got shape {value.shape}'

def test_train():
    env=EnvTwoKings()
    n_batches=3
    n_games_per_batch=2
    buffer_size=5
    batch_size=2
    n_simulations=3
    learning_rate=0.01
    c_weight_decay=0.0
    c_puct=0.1
    temp=1.0
    alpha_dir=1.0
    eps_dir=0.5

    losses, losses_pol_black, losses_pol_white, losses_val, win_frac = train(
        env=env,
        net=net,
        n_batches=n_batches,
        n_games_per_batch=n_games_per_batch,
        buffer_size=buffer_size,
        batch_size=batch_size,
        n_simulations=n_simulations,
        learning_rate=learning_rate,
        c_weight_decay=c_weight_decay,
        c_puct=c_puct,
        alpha_dir=alpha_dir,
        eps_dir=eps_dir,
        temp=temp
    )

    assert len(losses) == n_batches
    assert len(losses_pol_black) == n_batches
    assert len(losses_pol_white) == n_batches
    assert len(losses_val) == n_batches
    assert len(win_frac) == n_batches







