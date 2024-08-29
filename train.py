from network import Network
from protocols import EnvProtocol
from mcts import mcts

import torch
import random
from collections import deque
from tqdm import tqdm


def self_play_game(env: EnvProtocol, net: Network) -> list[tuple]:
    """Network net plays game in environment env against itself.
    
    :param env: environment
    :param net: network
    :return buffer: list of (state, action, value) tuples where:
        state: 4D tensor corresponding to state of the game
        action: 4D tensor corresponding to action taken in the game
        value: 2D tensor: [[+1]], [[-1]], or [[0]] for white W, black W, draw
    """

    result = None
    state = env.state
    buffer = []

    # game loop
    while not result:
        action = mcts(env, net, n_simulations)
        buffer.append((state, action))
        state, result = env.step(action)

    # add ground truth value (ie outcome of game) to items in buffer
    if result == 'white':
        value = torch.tensor([[1]], dtype=torch.float)
    elif result == 'draw':
        value = torch.tensor([[0]], dtype=torch.float)
    elif result == 'black':
        value = torch.tensor([[-1]], dtype=torch.float)
    else:
        raise RuntimeError(f"result should be 'white', 'black', or 'draw' but got {result}")

    buffer = [(state, action, value) for state, action in buffer]

    return buffer




def train(
        net: network.Network
):
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

    for batch in tqdm(range(n_batches)):

        # Collect data via self-play
        for _ in range(n_games_per_batch):
            buffer.extend(self_play_game())

        # Use data to train model
        buffer_sample = random.sample(buffer, batch_size)

        state = torch.cat([item[0] for item in buffer_sample])
        action = torch.cat([item[1] for item in buffer_sample])
        value = torch.cat([item[2] for item in buffer_sample])

        logits, value_pred = net(state)

        loss_pol = loss_fn_pol(logits, action)
        loss_val = loss_fn_val(value_pred, value)
        loss = loss_pol + loss_val

        losses_pol.append(loss_pol.item())
        losses_val.append(loss_val.item())
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save checkpoint
        if (batch + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_folder, checkpoint_prefix, batch + 1)

        # Print progress
        if batch % (n_batches // 10) == 0:
            print(f'Batch {batch:3d}: loss_pol = {loss_pol.item():5.2f}, loss_val = {loss_val.item():5.2f}')

    return losses, losses_pol, losses_val

        





