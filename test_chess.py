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

    
def test_ResBlock():
    batch_size = 16
    num_filters = 8
    board_size = 5
    kernel_size = 3

    block = chess.ResBlock(
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

    policy = chess.PolicyHead(
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

    value = chess.ValueHead(
        board_size=board_size,
        num_filters=num_filters,
        hidden_layer_size=hidden_layer_size
    )

    x = torch.rand(batch_size, num_filters, board_size, board_size)
    y = value(x)

    assert y.shape == torch.Size([batch_size])


def test_Network():

    batch_size = 16
    params = {
        'in_channels': 4, 
        'board_size': 5,
        'num_filters': 8, 
        'kernel_size': 3, 
        'num_res_blocks': 6,
        'num_policy_filters': 2,
        'num_out_channels': 4,
        'value_hidden_layer_size': 64
    }

    network = chess.Network(**params)

    x = torch.rand(batch_size, params['in_channels'], 
                   params['board_size'], params['board_size'])
    
    action_dist, value = network(x)

    assert action_dist.shape == torch.Size([batch_size, params['num_out_channels'], params['board_size'], params['board_size']])

    assert value.shape == torch.Size([batch_size])
