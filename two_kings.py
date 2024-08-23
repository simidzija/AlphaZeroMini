import torch

# Valid actions mask: input state -> mask containing valid actions
def action_mask(state: torch.BoolTensor) -> torch.BoolTensor:
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

    def step(self, action: torch.IntTensor) -> tuple:
        """Update env based on action.
        
        :param action: int tensor [[dir, row, col]] (batch size of 1)
        :return: tuple (state, result) where:
            state: tensor of shape (1,3,5,5)
            result: "white", "black", "draw" or None (if game not over)
        """
        assert action.shape == torch.Size([1,3])
        assert action.dtype == torch.int
        
        # move P1
        P1 = self.state[0]
        P2 = self.state[1]

        direction, row, col = action.squeeze()
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
        if torch.all(P1 == P2):
            P2[row, col] = 0 # "take P2's king"
            return self.state, self.color
        # check if move limit reached
        elif self.color == 'black' and self.move == self.move_limit:
            return self.state, "draw"
        # otherwise play on: rotate board, change color, increase move count
        else:
            self.state[0], self.state[1] = self.state[1].flip(0,1), self.state[0].flip(0,1)
            if self.color == 'black':
                self.move += 1
                self.color = 'white'
            else:
                self.color = 'black'
            return self.state, None


