import torch

def action_dist(logits: torch.Tensor):
    """Represent policy logits as probability distribution over actions.
    
    :param logits: tensor of shape (1,o,s,s) (batch size of 1)
    :return actions, probs: 
        actions: tensor of shape (n,1,3)
        probs: list of ints of length n
        
    Where n is the number of non-zero entries in policy
    """
    logits = logits.squeeze(0) # get rid of batch dim
    probs_tensor = logits.flatten().softmax(dim=0).reshape(logits.shape)
    actions = probs_tensor.nonzero().unsqueeze(1).to(torch.int)
    probs = probs_tensor.flatten()[probs_tensor.flatten() != 0].tolist()

    return actions, probs

