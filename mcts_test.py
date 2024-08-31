import mcts
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

def test_Tree():
    env = EnvTwoKings()
    tree = mcts.Tree(env, net, c_puct=0.1, temp=1)

    tree.simulation()
    assert len(tree.root.children) == 3
    assert sum(edge['N'] for edge in tree.root.children.values()) == 0

    tree.simulation()
    assert len(tree.root.children) == 3
    assert sum(edge['N'] for edge in tree.root.children.values()) == 1
    # find the child node that got visited
    for child, edge in tree.root.children.items():
        if edge['N'] == 1:
            break
    policy, value = net(child.state)
    value = value.item()
    assert tree.root.children[child]['W'] == value
    assert tree.root.children[child]['Q'] == value

    action = tree.get_action()
    assert action.shape == torch.Size([1,3])
    
def test_mcts():
    env = EnvTwoKings()
    action = mcts.mcts(env, net, n_simulations=10, c_putc=0.1, temp=1)

    assert action.shape == torch.Size([1,3])
    assert 0 <= action[0,0] < 4
    assert action[0,1] == 4, f'row index of king in starting position should be 4 but got {action[0,1]}'
    assert action[0,2] == 2, f'col index of king in starting position should be 2 but got {action[0,2]}'



