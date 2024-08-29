from network import Network
from protocols import EnvProtocol

import torch
import copy


class Node:
    def __init__(self, state: torch.BoolTensor):
        self.state = state
        self.children = {}

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return bool(torch.all(self.state == other.state))

    def __hash__(self):
        return hash(self.state.cpu().numpy().tobytes())

n1 = Node(torch.tensor([1,2], dtype=torch.bool))
n2 = Node(torch.tensor([1,2], dtype=torch.bool))

d = {}
d[n1] = 3
print(d[n2])




class Tree:
    def __init__(self, state: torch.BoolTensor):
        self.root = Node(state)

    def simulation(self):
        pass

    def get_action(self, temp):
        pass
        


def mcts(env: EnvProtocol, net: Network, n_simulations: int, temp: float) -> torch.IntTensor:
    """Monte Carlo Tree Search in environment env using network net.
    
    :param env: environment
    :param net: network
    :param n_simulations: number of simulations
    :param temp: temperature parameter for selecting final action
    :return action: 4D tensor (b.s. = 1) corresponding to action chosen by MCTS
    """
    tree = Tree()

    for _ in range(n_simulations):
        sim_env = copy.deepcopy(env)
        tree.simulation()

    return tree.get_action(temp)



