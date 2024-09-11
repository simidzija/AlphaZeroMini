from network import Network
from utils import action_dist
from protocols import EnvProtocol

import torch
import math
from typing import Optional
import random

class Node:
    def __init__(self, state: torch.BoolTensor, result=None):
        self.state = state
        self.result = result
        if result is None:
            self.children = {}
        else:
            self.children = None
        self.sum_N = 0 # sum of N values of all edges starting at node

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return bool(torch.all(self.state == other.state))

    def __hash__(self):
        return hash(self.state.cpu().numpy().tobytes())

class Tree:
    def __init__(self, env: EnvProtocol, net: Network, c_puct: float, temp: float, new_root: Optional[Node]=None):
        self.env = env
        self.net = net
        self.c_puct = c_puct
        self.temp = temp
        if new_root is None:
            self.root = Node(env.state)
        elif isinstance(new_root, Node):
            assert bool(torch.all(new_root.state == env.state)), f'state of new_root must match state of env'
            self.root = new_root

    def simulation(self):
        current = self.root
        path = [current]

        # go from root to leaf
        while current.children:
            max_Q_plus_U = -float('inf')
            for child in current.children:
                edge = current.children[child]
                P = edge['P']
                N = edge['N']
                Q = edge['Q']
                sum_N = current.sum_N
                if current.sum_N == 0:
                    # the first time we explore actions from a node just sample using prior probs
                    U = P
                else:
                    # otherwise use the PUCT formula from AlphaGo Zero appendix
                    U = self.c_puct * P * math.sqrt(sum_N) / (1 + N)

                if Q + U > max_Q_plus_U:
                    max_Q_plus_U = Q + U
                    next_node = child
            path.append(next_node)
            current = next_node
        
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
                # define edge from current to leaf
                current.children[leaf] = {
                    'action': action, # action
                    'P': prob, # prior prob
                    'N': 0, # visit count
                    'W': 0, # total action value
                    'Q': 0 # mean action value
                }
        elif current.result == 'white' or current.result == 'black':
            value = +1 if self.env.color == current.result else -1
        elif current.result == 'draw':
            value = 0
        else:
            raise RuntimeError(f'Got invalid result: {current.result}')

        # backup
        if len(path) > 1:
            for parent, child in zip(path[:-1], path[1:]):
                edge = parent.children[child]
                edge['N'] += 1
                edge['W'] += value
                edge['Q'] = edge['W'] / edge['N']
                parent.sum_N += 1

    def get_action(self):
        """Sample action based on weights given by N^(1/T) where T is temp.
        
        :return: tuple (action, new_root) where:
            action: tensor [[dir, row, col]] corresponding to action taken
            new_root: Node object resulting from action
        """
        if not self.root.children:
            raise RuntimeError('Cannot call get_action() on Tree with childless root node.')

        children = []
        weights = []
        for child, edge in self.root.children.items():
            children.append(child)
            weights.append(edge['N'] ** (1 / self.temp))

        new_root = random.choices(children, weights)[0]
        action = self.root.children[new_root]['action']

        return action, new_root


def mcts(tree: Tree, n_simulations: int) -> torch.IntTensor:
    """Monte Carlo Tree Search
    
    :param tree: starting tree
    :param n_simulations: number of simulations

    :return: tuple (action, new_root) where:
        action: tensor [[dir, row, col]] corresponding to action taken
        new_root: Node object resulting from action
    """
    for _ in range(n_simulations):
        tree.simulation()

    action, new_root = tree.get_action()
    return action, new_root

