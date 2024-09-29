"""
Training a neural network to play a board game.

This module defines methods for training a neural network to play a board game, 
defined by specifying a game environment. The methods are:
    - self_play_game: method for neural net to play against itself
    - train: method for training a neural net to become better at a board game.
      Uses the mcts module, for Monte Carlo Tree Search.
"""


from network import Network
from protocols import EnvProtocol
from two_kings import EnvTwoKings, action_mask
from mcts import Tree, mcts

import torch
import random
from collections import deque
from typing import Optional
import os
import matplotlib.pyplot as plt


def self_play_game(env: EnvProtocol, 
                   net: Network, 
                   n_simulations: int,
                   c_puct: float,
                   temp: float,
                   alpha_dir: float,
                   eps_dir: float,
                   print_move: Optional[bool]=False) -> list[tuple]:
    """Network net plays game in environment env against itself.
    
    :param env: environment
    :param net: network
    :param n_simulations: number of simulations per MCTS

    :return buffer: list of (state, action, value) tuples where:
        state: 4D tensor (bs = 1) corresponding to state of the game
        action: tensor [[dir, row, col]] corresponding to action taken
        value: tensor: [[-1]], [[+1]], or [[0]] for white W, black W, draw
    """

    result = None
    state = env.state.clone()
    buffer = []

    if print_move:
        print('  ', end='')

    # game loop
    new_root = None
    while not result:
        # initialize tree
        tree = Tree(env, net, c_puct, temp, alpha_dir, eps_dir, new_root)
        # get action and new root Node
        action, new_root = mcts(tree, n_simulations=n_simulations)
        # print(tree)
        # append (state, action) to buffer
        buffer.append((state, action))
        # step in env
        state, result = env.step(action, print_move=print_move)

    # add ground truth value (ie outcome of game) to items in buffer
    if result == 'white':
        value = torch.tensor([[-1]], dtype=torch.float)
    elif result == 'draw':
        value = torch.tensor([[0]], dtype=torch.float)
    elif result == 'black':
        value = torch.tensor([[+1]], dtype=torch.float)
    else:
        raise RuntimeError(f"result should be 'white', 'black', or 'draw' but got {result}")

    buffer = [(state, action, value) for state, action in buffer]

    return buffer


def train(env: EnvProtocol,
          net: Network,
          *,
          n_batches: int,
          n_games_per_batch: int,
          buffer_size: int,
          batch_size: int,
          n_simulations: int,
          learning_rate: float,
          c_weight_decay: float,
          c_puct: float,
          temp: float,
          alpha_dir: float,
          eps_dir: float,
          checkpoint_interval: Optional[int]=None):
    """Train a network in an environment by allowing it to play against itself.
    
    :param env: the nvironment
    :param net: the network
    :param n_batches: number of training batches
    :param n_games_per_batch: number of self-play games per batch
    :param batch_size: batch size (number of board states)
    :param n_simulations: number of MCTS simulations per every move
    :param learning_rate: initial learning rate for gradient descent
    :param c_weight_decay: weight decay constant
    :param c_puct: PUCT constant; see appendix of AGZ paper for details
    :param temp: temperature for sampling moves following MCTS simulations
    :param alpha_dir: Dirichlet noise alpha parameter
    :param eps_dir: Dirichlet noise epsilon parameter (overall noise scale)
    :param checkpoint_interval: period of batches between saving checkpoints

    :return: lists with batch statistics
        - losses_tot: total losses
        - losses_pol_black: CE losses from policy head on black's moves
        - losses_pol_white: CE losses from policy head on white's moves
        - losses_val: MSE losses from value head
        - win_frac_list: fraction of training games that resulted in a win for 
          either player    
    """
        
    assert buffer_size >= batch_size
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, 
                                 weight_decay=c_weight_decay)
    loss_fn_pol = torch.nn.CrossEntropyLoss() 
    loss_fn_val = torch.nn.MSELoss()

    buffer = deque([], maxlen=buffer_size)  # items (state, action, value)

    # logging
    losses_tot = []
    losses_pol_black = []
    losses_pol_white = []
    losses_val = []
    win_frac_list = []

    for batch in range(n_batches):

        print_freq = 1
        if batch % print_freq == 0:
            print(f'Batch {batch}:')

        # Collect data via self-play
        for _ in range(n_games_per_batch):
            env_clone = env.clone()
            buffer.extend(self_play_game(env_clone, 
                                         net, 
                                         n_simulations=n_simulations, 
                                         c_puct=c_puct, 
                                         temp=temp, 
                                         alpha_dir=alpha_dir,
                                         eps_dir=eps_dir,
                                         print_move=True))

        buffer_sample = random.sample(buffer, batch_size)

        state = torch.cat([item[0] for item in buffer_sample])
        action = torch.cat([item[1] for item in buffer_sample])
        value = torch.cat([item[2] for item in buffer_sample])

        # Statistics of training values (wins vs draws)
        win_frac = value.flatten().tolist().count(1) / batch_size
        win_frac_list.append(win_frac)

        # Predict actions and values
        logits, value_pred = net(state)

        logits_flat = logits.flatten(1)
        action_flat = (action[:, 0] * net.board_size ** 2 + 
                       action[:, 1] * net.board_size + 
                       action[:, 2]) # converts 3D index to 1D index
        action_flat = action_flat.to(torch.long)

        # Color masks
        black_mask = state[:, 2, 0, 0] == 1
        white_mask = state[:, 2, 0, 0] == -1

        # Compute losses
        # loss_pol = loss_fn_pol(logits_flat, action_flat)
        logits_black = logits_flat[black_mask]
        logits_white = logits_flat[white_mask]
        action_black = action_flat[black_mask]
        action_white = action_flat[white_mask]
        # print(f'logits_black: {logits_black}')
        # print(f'action_black: {action_black}')
        # print(f'logits_white: {logits_white}')
        # print(f'action_white: {action_white}')

        loss_pol_black = loss_fn_pol(logits_black, action_black)
        loss_pol_white = loss_fn_pol(logits_white, action_white)
        loss_val = loss_fn_val(value_pred, value)

        loss_tot = loss_pol_black + loss_pol_white + loss_val

        # losses_pol.append(loss_pol.item())
        losses_pol_black.append(loss_pol_black.item())
        losses_pol_white.append(loss_pol_white.item())
        losses_val.append(loss_val.item())
        losses_tot.append(loss_tot.item())

        # Backprop and gradient descent
        optimizer.zero_grad()
        loss_tot.backward()
        optimizer.step()

        # Save checkpoint
        if checkpoint_interval and (batch + 1) % checkpoint_interval == 0:
            filename = os.path.join('checkpoints', f'batch_{batch + 1}.pth')
            torch.save(net, filename)

        # Print progress
        if batch % print_freq == 0:
            print(f'  loss_tot = {loss_tot.item():5.2f}')
            print(f'  loss_pol_black = {loss_pol_black.item():7.4f}')
            print(f'  loss_pol_white = {loss_pol_white.item():7.4f}')
            print(f'  loss_val = {loss_val.item():7.4f}')
            print(f'  black win_frac = {win_frac:5.2f}')

    return losses_tot, losses_pol_black, losses_pol_white, losses_val, win_frac_list

        
if __name__ == '__main__':

    env = EnvTwoKings()
    net = Network(
        num_in_channels=4,
        board_size=5,
        num_filters=8,
        kernel_size=1,
        num_res_blocks=6,
        num_policy_filters=2,
        num_out_channels=4,
        value_hidden_layer_size=32,
        action_mask=action_mask
    )

    losses, losses_pol_black, losses_pol_white, losses_val, win_frac_list = train(
        env=env,
        net=net,
        n_batches=100,
        n_games_per_batch=5,
        buffer_size=50, # should be < games * (least possible states/game)
        batch_size=50, # should be <= buffer_size
        n_simulations=200,
        learning_rate=0.02,
        c_weight_decay=0.0,
        c_puct=0.5, # higher value favours prior probabilities
        temp=1.0, # higher value means more exploration
        alpha_dir=0.5, # higher value means more exploration
        eps_dir=0.25, # higher value means more exploration
        checkpoint_interval=10 # should be an O(1) fraction of n_batches
    )

    plt.plot(losses, label='loss')
    plt.plot(losses_pol_black, label='loss_pol_black')
    plt.plot(losses_pol_white, label='loss_pol_white')
    plt.plot(losses_val, label='loss_val')
    plt.yscale('log')
    plt.legend()
    plt.show()








