from network import Network, action_dist
from protocols import EnvProtocol

import torch

class Node:
    def __init__(self, state: torch.BoolTensor, result=None):
        self.state = state
        self.result = result
        if result is None:
            self.children = {}

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return bool(torch.all(self.state == other.state))

    def __hash__(self):
        return hash(self.state.cpu().numpy().tobytes())

class Tree:
    def __init__(self, env: EnvProtocol, net: Network):
        self.env = env
        self.net = net
        self.root = Node(env.state)

    def simulation(self):
        current = self.root
        path = [current]

        # go from root to leaf
        while current.children:
            pass
        
        # expand leaf
        if current.result is None:
            # create new env with current state
            env = self.env.new_env(state=current.state)

            # get policy and value
            logits, value = self.net(env.state)
            value = value.item()

            # get action distribution
            actions, probs = action_dist(logits)

            # create new leaves
            for action, prob in zip(actions, probs):
                state, result = env.step(action, update_state=False)
                leaf = Node(state=state, result=result)
                current.children[leaf] = {
                    'action': action, # action
                    'P': prob, # prior prob
                    'N': 0, # visit count
                    'W': 0, # total action value
                    'Q': 0 # mean action value
                }
        elif current.result == 'white':
            value = -1
        elif current.result == 'black':
            value = +1
        elif current.result == 'draw':
            value = 0
        else:
            raise RuntimeError(f'Got invalid result: {result}')

        # backup
        if len(path) > 1:
            for parent, child in zip(path[:-1], path[1:]):
                edge = parent.children[child]
                edge['N'] += 1
                edge['W'] += value
                edge['Q'] = edge['W'] / edge['N']

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
        tree.simulation()

    return tree.get_action(temp)



