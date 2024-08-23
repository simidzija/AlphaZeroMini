import torch
from torch import nn
from torch.distributions import Categorical

# Valid actions mask: input state -> mask containing valid actions
def action_mask_two_kings(state: torch.BoolTensor) -> torch.BoolTensor:
    """Mask of allowed actions.

    :param state: bx3x5x5 tensor. Each batch b has 3 5x5 planes: P1 pos, P2 pos, move number
    :return mask: bx4x5x5 tensor. Each batch b has 4 5x5 planes: move up, down, left, right
    """
    # TODO: Figure out a more efficient / "torchy" implementation

    num_batches = state.size(0)
    mask = torch.zeros(num_batches, 4, 5, 5, dtype=torch.bool)

    for b in range(num_batches):
        row, col = state[b, 0].nonzero(as_tuple=True)

        if row > 0:
            mask[b, 0, row, col] = 1
        if row < 4:
            mask[b, 1, row, col] = 1
        if col > 0:
            mask[b, 2, row, col] = 1
        if col < 4:
            mask[b, 3, row, col] = 1
    
    return mask

# Policy and value network:
class ResBlock(nn.Module):
    """Two convolution layers with a residual skip connection."""
    def __init__(self, num_filters: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, f'kernel size must be odd, got {kernel_size}'
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=padding
        )
        self.bn1 = nn.BatchNorm2d(num_features=num_filters)
        self.bn2 = nn.BatchNorm2d(num_features=num_filters)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = x + y
        y = self.relu(y)

        return y
    
class PolicyHead(nn.Module):
    """Policy head."""
    def __init__(self, board_size: int, num_filters:int, num_policy_filters: int, num_out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, 
                      out_channels=num_policy_filters, 
                      kernel_size=1),
            nn.BatchNorm2d(num_features=num_policy_filters),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(in_features=num_policy_filters * board_size ** 2,
                      out_features=num_out_channels * board_size ** 2),
            nn.Unflatten(1, (num_out_channels, board_size, board_size))
        )

    def forward(self, x):
        return self.layers(x)

class ValueHead(nn.Module):
    """Value Head."""
    def __init__(self, board_size: int, num_filters: int, hidden_layer_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(in_features=board_size**2, 
                      out_features=hidden_layer_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_layer_size, out_features=1),
            nn.Tanh(),
            nn.Flatten(0)
        )

    def forward(self, x):
        return self.layers(x)

class Network(nn.Module):
    """Policy and value network."""
    def __init__(self, *, 
                 num_in_channels: int, 
                 board_size: int,
                 num_filters: int, 
                 kernel_size:int, 
                 num_res_blocks: int,
                 num_policy_filters: int,
                 num_out_channels: int,
                 value_hidden_layer_size: int,
                 action_mask
                 ):
        super().__init__()

        self.action_mask = action_mask

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_in_channels,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList([ResBlock(num_filters, kernel_size) for _ in range(num_res_blocks)])

        self.policy_head = PolicyHead(
            board_size=board_size,
            num_filters=num_filters,
            num_policy_filters=num_policy_filters,
            num_out_channels=num_out_channels
        )

        self.value_head = ValueHead(
            board_size=board_size,
            num_filters=num_filters,
            hidden_layer_size=value_hidden_layer_size
        )

    def forward(self, state: torch.BoolTensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        :param state: tensor of shape (b,i,s,s) where
        :return: tuple (logits, value) where
            logits.shape = (b,o,s,s) 
            value.shape = (b,)

        And where: 
            b = batch size
            i = input channels
            o = output channels
            s = board size
        """
        y = self.block1(state)
        for block in self.res_blocks:
            y = block(y)
        
        # policy
        unmasked = self.policy_head(y)
        mask = self.action_mask(state)
        logits = torch.where(mask, unmasked, -torch.inf)

        # value
        value = self.value_head(y)

        return logits, value

    def greedy_sample(self, state: torch.BoolTensor) -> torch.Tensor:
        """Take state, produce action dist, sample greedily from dist.
        
        :param state: tensor of shape (b,i,s,s)
        :return: tensor of shape (b,3) corresponding to 3 indices (output plane, row, column)

        Where:
            b = batch size
            i = input channels
            s = board size
        """
        logits, _ = self(state)
        dist = Categorical(logits=logits.flatten(1)) # dist over non-batch dims
        actions = dist.sample()
        shape = logits[0].shape
        actions = torch.stack(torch.unravel_index(actions, shape)).T

        return actions

    
# Environment
class EnvTwoKings:
    """Environment for Two Kings game."""
    def __init__(self):
        # Note: no batching
        self.state = torch.zeros(3, 5, 5, dtype=torch.bool)
        self.state[0, 4, 2] = 1 # white (P1) at c1
        self.state[1, 0, 2] = 1 # black (P2) at c5
        self.color = 'white'
        self.move = 1
        self.move_limit = 10

    def step(self, action: torch.Tensor) -> tuple:
        """Update env based on action.
        
        :param action: tensor [[dir, row, col]] (batch size of 1)
        :return: tuple (state, result) where:
            state: tensor of shape (1,3,5,5)
            result: "white", "black", "draw" or None (if game not over)
        """
        assert action.shape == torch.Size([1,3])
        
        # move P1
        P1 = self.state[0]
        P2 = self.state[1]

        direction, row, col = action.unsqueeze()
        P1[row, col] = 0 # "pick up piece"

        if direction == 0: # up
            row -= 1
        elif direction == 1: # down
            row += 1
        elif direction == 2: # left
            col -= 1
        elif direction == 3: # right
            col += 1
        else:
            raise RuntimeError(f'direction must be 0,1,2,3 but got {direction}')
        
        P1[row, col] = 1 # "put down piece"

        # check if P1 won
        if P1 == P2:
            P2[row, col] = 0 # "take P2's king"
            return self.state, self.color
        # check if move limit reached
        elif self.color == 'black' and self.move == self.move_limit:
            return self.state, "draw"
        # otherwise play on: flip board, change color, increase move count
        else:
            self.state[0], self.state[1] = self.state[1], self.state[0].clone()
            self.color = 'black' if self.color == 'white' else 'white'
            self.move += 1
            return self.state, self.color






# Implement GUI which lets me play :)



# MCTS (think about how to structure this)



# Training function



# Save final model to disk



# Load and play trained model