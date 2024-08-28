import network
import torch
import random
from collections import deque
from tqdm import tqdm


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
            buffer.extend(play_game())

        # Use data to train model
        buffer_sample = random.sample(buffer, batch_size)

        state = torch.stack([item[0] for item in buffer_sample])
        action = torch.stack([item[1] for item in buffer_sample])
        value = torch.stack([item[2] for item in buffer_sample])

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

        





