import torch
from torch import nn
from torch.distributions import Categorical

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
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)

class Network(nn.Module):
    """Policy and value network."""
    def __init__(self,
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

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        :param state: tensor of shape (b,i,s,s) where
        :return: tuple (logits, value) where
            logits.shape = (b,o,s,s) 
            value.shape = (b,1)

        And where: 
            b = batch size
            i = input channels
            o = output channels
            s = board size
        """

        # ensure state is 4D (ie batched)
        if state.ndim == 4:
            pass
        elif state.ndim == 3:
            state = state.unsqueeze(0)
        else:
            raise ValueError(f'state must be 3D or 4D tensor, but got {state.ndim}D tensor')

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

    def greedy_sample(self, state: torch.BoolTensor) -> torch.IntTensor:
        """Take state, produce action dist, sample greedily from dist.
        
        :param state: tensor of shape (b,i,s,s)
        :return action: tensor of shape (b,3) corresponding to 3 indices (output plane, row, column)

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

        return actions.to(torch.int)


# MCTS (think about how to structure this)



# Training function



# Save final model to disk



# Load and play trained model