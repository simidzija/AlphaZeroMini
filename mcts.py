from network import Network
from utils import action_dist
from protocols import EnvProtocol

import torch
from torch.distributions.dirichlet import Dirichlet
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
    
    @property
    def color(self):
        if self.state[0, 2, 0, 0] == -1:
            return 'white'
        elif self.state[0, 2, 0, 0] == 1:
            return 'black'
        else:
            raise RuntimeError(f'self.state[0, 2] must have all elements equal to 0 or 1, but got {self.state[0, 2]}')
        
    @property
    def move_num(self):
        return int(self.state[0, 3, 0, 0].item())
    
class Tree:
    def __init__(self, env: EnvProtocol, net: Network, c_puct: float, temp: float, alpha_dir: float, eps_dir: float, new_root: Optional[Node]=None):
        self.env = env
        self.net = net
        self.c_puct = c_puct
        self.temp = temp
        self.alpha_dir = alpha_dir
        self.eps_dir = eps_dir
        if new_root is None:
            self.root = Node(env.state)
        elif isinstance(new_root, Node):
            assert bool(torch.all(new_root.state == env.state)), f'state of new_root must match state of env'
            self.root = new_root
    
    def __str__(self):
        def edge_and_node(edge: dict, node: Node, indent: int) -> str:
            """Prints edge and child node."""
            
            action = edge['action']
            # node represents the child of the "current node" so must adjust color and move_num accordingly
            if node.color == 'white':
                color = 'black'
                move_num = node.move_num - 1
            elif node.color == 'black':
                color = 'white'
                move_num = node.move_num

            string = " " * indent
            string += "B " if color == 'black' else "W "
            string += "--- ("
            string += f"{edge['N']:2d}, "
            string += f"{edge['P']:.2f}, "
            string += f"{edge['W']:.2f}, "
            string += f"{edge['Q']:.2f}"
        
            # move
            move = self.env.get_move(action, color)
            string += f") --- {move_num}."
            string += " ... " if color == 'black' else ""
            string += f"{move}"

            # result (if simulation reached end of game)
            result = node.result
            if result == 'black':
                string += " BLACK WIN\n"
            elif result == 'white':
                string += " WHITE WIN\n"
            elif result == 'draw':
                string += " DRAW\n" 
            else:
                string += "\n"
            
            return string
            
        # edges are labelled (N,P,W,Q)
        string = f'\n\nRoot Node\n'
        
        stack = [(edge, node, 4) for node, edge in self.root.children.items()]
        while stack:
            edge, node, indent = stack.pop()
            string += edge_and_node(edge, node, indent)
            if not node.children:
                continue
            for child, edge in node.children.items():
                stack.append((edge, child, indent + 4))
        
        return string

    def simulation(self):
        current = self.root
        path = [current]

        # go from root to leaf
        while current.children:
            # Take path which maximizes Q_color + U, where Q_color is Q when the current player is black, and -Q if white. 
            max_Q_plus_U = -float('inf')
            for child in current.children:
                edge = current.children[child]
                P = edge['P']
                N = edge['N']
                Q_color = edge['Q'] if current.color == 'black' else -edge['Q']
                sum_N = current.sum_N
                if current.sum_N == 0:
                    # the first time we explore actions from a node just take action with highest prior
                    U = P
                else:
                    # otherwise use the PUCT formula from AlphaGo Zero appendix
                    U = self.c_puct * P * math.sqrt(sum_N) / (1 + N)

                # print(f'Q_color: {Q_color:.4f}, U = {U:.4f}, Q_color+U: {Q_color + U:.4f}')
                if Q_color + U > max_Q_plus_U:
                    max_Q_plus_U = Q_color + U
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

            # Dirichlet noise
            if current is self.root:
                concentration = torch.full((len(actions),), self.alpha_dir)
                dir_dist = Dirichlet(concentration)
                dir_sample = dir_dist.sample().tolist()
            else:
                dir_sample = [0] * len(actions)

            # create new leaves
            for action, prob, dir_noise in zip(actions, probs, dir_sample):
                state, result = env.step(action, update_state=False)
                leaf = Node(state=state, result=result)
                # define edge from current to leaf
                current.children[leaf] = {
                    'action': action, # action
                    'P': (1 - self.eps_dir) * prob + self.eps_dir * dir_noise, # prior prob
                    'N': 0, # visit count
                    'W': 0, # total action value
                    'Q': 0 # mean action value
                }
        elif current.result == 'black':
            value = +1
            # print(f"\nWin detected for black")
        elif current.result == 'white':
            value = -1
            # print(f"\nWin detected for white")
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

        # print(self)

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

