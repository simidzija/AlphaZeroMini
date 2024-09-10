from network import Network
from utils import action_dist
from protocols import EnvProtocol

import torch
import math
from typing import Any
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
    def __init__(self, env: EnvProtocol, net: Network, c_puct: float, temp: float):
        self.env = env
        self.net = net
        self.c_puct = c_puct
        self.temp = temp
        self.root = Node(env.state)

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
        
        :return action: tensor of shape (1,3)
        """
        if not self.root.children:
            raise RuntimeError('Cannot call get_action() on Tree with childless root node.')
        actions = []
        weights = []
        for edge in self.root.children.values():
            actions.append(edge['action'])
            weights.append(edge['N'] ** (1 / self.temp))

        action = random.choices(actions, weights)[0]

        return action


def mcts(env: EnvProtocol, net: Network, n_simulations: int, 
         c_putc: float, temp: float) -> torch.IntTensor:
    """Monte Carlo Tree Search in environment env using network net.
    
    :param env: environment
    :param net: network
    :param n_simulations: number of simulations
    :param temp: temperature parameter for selecting final action

    :return action: 4D tensor (b.s. = 1) corresponding to action chosen by MCTS
    """
    tree = Tree(env, net, c_putc, temp)

    for _ in range(n_simulations):
        tree.simulation()

    action = tree.get_action()
    return action



