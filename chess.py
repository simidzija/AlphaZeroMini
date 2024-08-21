import torch


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



# Implement GUI which lets me play :)



# MCTS (think about how to structure this)



# Training function



# Save final model to disk



# Load and play trained model