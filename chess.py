import torch
from torch import nn


# Valid actions mask: input features -> mask containing valid actions
def action_mask(features: torch.BoolTensor) -> torch.BoolTensor:
    """Mask of allowed actions.

    :param features: 4x5x5 tensor corresponding to white pos, black pos, color, move count
    :return mask: 4x5x5 mask of allowed actions corresponding to up, down, left, right
    """
    mask = torch.zeros(4, 5, 5, dtype=torch.bool)
    black = features[2,0,0].item()
    row, col = (features[1] if black else features[0]).nonzero(as_tuple=True)

    if row > 0:
        mask[0, row, col] = 1
    if row < 4:
        mask[1, row, col] = 1
    if col > 0:
        mask[2, row, col] = 1
    if col < 4:
        mask[3, row, col] = 1
    
    return mask



# Policy and value network: input features -> prob dist over valid actions 
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
                 in_channels: int, 
                 board_size: int,
                 num_filters: int, 
                 kernel_size:int, 
                 num_res_blocks: int,
                 num_policy_filters: int,
                 num_out_channels: int,
                 value_hidden_layer_size: int
                 ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
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


    def forward(self, x):
        y = self.block1(x)
        for block in self.res_blocks:
            y = block(y)
        
        policy = self.policy_head(y)
        value = self.value_head(y)

        return policy, value



# Implement GUI which lets me play :)



# MCTS (think about how to structure this)



# Training function



# Save final model to disk



# Load and play trained model