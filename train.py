from network import Network
from protocols import EnvProtocol
from two_kings import EnvTwoKings, action_mask
from mcts import mcts

import torch
import random
from collections import deque
from tqdm import tqdm
from typing import Optional
import os
import matplotlib.pyplot as plt


def self_play_game(env: EnvProtocol, 
                   net: Network, 
                   n_simulations: int,
                   c_putc: float,
                   temp: float,
                   print_move: Optional[bool]=False) -> list[tuple]:
    """Network net plays game in environment env against itself.
    
    :param env: environment
    :param net: network
    :param n_simulations: number of simulations per MCTS

    :return buffer: list of (state, action, value) tuples where:
        state: 4D tensor (bs = 1) corresponding to state of the game
        action: tensor [[dir, row, col]] corresponding to action taken
        value: 1D tensor: [[-1]], [[+1]], or [[0]] for white W, black W, draw
    """

    result = None
    state = env.state.clone()
    buffer = []

    # game loop
    while not result:
        action = mcts(env, net, n_simulations=n_simulations, 
                      c_putc=c_putc, temp=temp)
        buffer.append((state, action))
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
          c_putc: float,
          temp: float,
          checkpoint_interval: Optional[int]=None):
        
    assert buffer_size > batch_size
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, 
                                 weight_decay=c_weight_decay)
    loss_fn_pol = torch.nn.CrossEntropyLoss() 
    loss_fn_val = torch.nn.MSELoss()

    buffer = deque([], maxlen=buffer_size)  # items (state, action, value)

    # logging
    losses = []
    losses_pol = []
    losses_val = []

    for batch in range(n_batches):

        # Collect data via self-play
        for _ in range(n_games_per_batch):
            env_clone = env.clone()
            print(f'  New game:')
            buffer.extend(self_play_game(env_clone, net, 
                                         n_simulations=n_simulations, 
                                         c_putc=c_putc, temp=temp,
                                         print_move=True))

        # Use data to train model
        buffer_sample = random.sample(buffer, batch_size)

        state = torch.cat([item[0] for item in buffer_sample])
        action = torch.cat([item[1] for item in buffer_sample])
        value = torch.cat([item[2] for item in buffer_sample])

        logits, value_pred = net(state)

        logits_flat = logits.flatten(1)
        action_flat = (action[:, 0] * net.board_size ** 2 + 
                       action[:, 1] * net.board_size + 
                       action[:, 2]) # converts 3D index to 1D index
        action_flat = action_flat.to(torch.long)

        loss_pol = loss_fn_pol(logits_flat, action_flat)
        loss_val = loss_fn_val(value_pred, value)
        loss = loss_pol + loss_val

        losses_pol.append(loss_pol.item())
        losses_val.append(loss_val.item())
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if checkpoint_interval and (batch + 1) % checkpoint_interval == 0:
            filename = os.path.join('checkpoints', f'batch_{batch + 1}.pth')
            torch.save(net, filename)

        # Print progress
        print_freq = 1
        if batch % print_freq == 0:
            print(f'Batch {batch:3d}: loss_pol = {loss_pol.item():5.2f}, loss_val = {loss_val.item():5.2f}')
            print(f'  value: {value}')
            print(f'  value_pred: {value_pred}')

    return losses, losses_pol, losses_val

        
if __name__ == '__main__':

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

    losses, losses_pol, losses_val = train(
        env=EnvTwoKings(),
        net=net,
        n_batches=30,
        n_games_per_batch=3,
        buffer_size=10,
        batch_size=3,
        n_simulations=5,
        learning_rate=0.01,
        c_weight_decay=0.1,
        c_putc=0.1,
        temp=1,
        checkpoint_interval=10
    )

    plt.plot(losses, label='loss')
    plt.plot(losses_pol, label='loss_pol')
    plt.plot(losses_val, label='loss_val')
    plt.yscale('log')
    plt.legend()
    plt.show()




