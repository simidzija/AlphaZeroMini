import network
import two_kings
import torch

def test_ResBlock():
    batch_size = 16
    num_filters = 8
    board_size = 5
    kernel_size = 3

    block = network.ResBlock(
        num_filters=num_filters,
        kernel_size=kernel_size
    )

    x = torch.rand(batch_size, num_filters, board_size, board_size)
    y = block(x)

    assert y.shape == x.shape


def test_PolicyHead():
    batch_size = 16
    board_size = 5
    num_filters = 8
    num_policy_filters = 2
    num_out_channels = 4

    policy = network.PolicyHead(
        board_size=board_size,
        num_filters=num_filters,
        num_policy_filters=num_policy_filters,
        num_out_channels=num_out_channels
    )

    x = torch.rand(batch_size, num_filters, board_size, board_size)
    y = policy(x)

    assert y.shape == torch.Size([batch_size, num_out_channels, board_size, board_size])


def test_ValueHead():
    batch_size = 16
    board_size = 5
    num_filters = 8
    hidden_layer_size = 64

    value = network.ValueHead(
        board_size=board_size,
        num_filters=num_filters,
        hidden_layer_size=hidden_layer_size
    )

    x = torch.rand(batch_size, num_filters, board_size, board_size)
    y = value(x)

    assert y.shape == torch.Size([batch_size, 1])


def test_Network():

    batch_size = 16
    params = {
        'num_in_channels': 3, 
        'board_size': 5,
        'num_filters': 8, 
        'kernel_size': 3, 
        'num_res_blocks': 6,
        'num_policy_filters': 2,
        'num_out_channels': 4,
        'value_hidden_layer_size': 64,
        'action_mask': two_kings.action_mask
    }

    net = network.Network(**params)

    state = torch.zeros(batch_size, params['num_in_channels'], 
                   params['board_size'], params['board_size'])
    state[:, 0, 3, 4] = 1 # P1 on (3,4) for all batches
    state[:, 1, 2, 1] = 1 # P2 on (2,1) for all batches
    state[:, 2, ...] = 7 # move 7 for all batches
    
    # forward()
    logits, value = net(state)

    assert logits.shape == torch.Size([batch_size, params['num_out_channels'], params['board_size'], params['board_size']])

    assert value.shape == torch.Size([batch_size, 1])

    # greedy_sample()
    s = net.greedy_sample(state)

    assert s.shape == torch.Size([batch_size, 3])

    assert torch.all(0 <= s[:, 0])
    assert torch.all(s[:, 0] < params['num_out_channels'])

    assert torch.all(0 <= s[:, 1])
    assert torch.all(s[:, 1] < params['board_size'])

    assert torch.all(0 <= s[:, 2])
    assert torch.all(s[:, 2] < params['board_size'])
    
def test_action_dist():
    num_out_channels = 4
    board_size = 5

    logits = torch.full((1, num_out_channels, board_size, board_size), -torch.inf)
    logits[0, 1, 2, 3] = 1
    logits[0, 2, 1, 4] = 2
    non_zero_entries = 2

    actions, probs = network.action_dist(logits)

    assert actions.shape == torch.Size([non_zero_entries, 1, 3]), f'got actions.shape = {actions.shape}'
    assert len(probs) == non_zero_entries


