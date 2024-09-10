import torch

def action_dist(logits: torch.Tensor):
    """Represent policy logits as probability distribution over actions.
    
    :param logits: tensor of shape (1,o,s,s) (batch size of 1)
    :return actions, probs: 
        actions: tensor of shape (n,1,3)
        probs: list of probabilities of length n
        
    Where n is the number of non-zero entries in policy
    """
    logits = logits.squeeze(0) # get rid of batch dim
    probs_tensor = logits.flatten().softmax(dim=0).reshape(logits.shape)
    actions = probs_tensor.nonzero().unsqueeze(1).to(torch.int)
    probs = probs_tensor.flatten()[probs_tensor.flatten() != 0].tolist()

    return actions, probs



from two_kings import EnvTwoKings, action_mask
from network import Network


# net = Network(
#     num_in_channels=4,
#     board_size=5,
#     num_filters=8,
#     kernel_size=3,
#     num_res_blocks=6,
#     num_policy_filters=2,
#     num_out_channels=4,
#     value_hidden_layer_size=32,
#     action_mask=action_mask
# )


net = torch.load('checkpoints/batch_100.pth')

env = EnvTwoKings()

logits, value = net(env.state)
actions, probs = action_dist(logits)

print(actions)
print(probs)




