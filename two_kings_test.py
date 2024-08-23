import two_kings
import torch

def test_action_mask():
    # test 1
    state = torch.zeros(1,3,5,5, dtype=torch.bool)
    state[0, 0, 2, 3] = 1 # P1 at (2,3)
    state[0, 1, 4, 4] = 1 # P2 at (4,4)
    state[0, 2, ...] = 7 # 7 moves played

    mask_idxs = two_kings.action_mask(state)[0].nonzero()
    expected_idxs = torch.tensor([[0, 2, 3],
                                  [1, 2, 3],
                                  [2, 2, 3],
                                  [3, 2, 3]])

    assert torch.all(mask_idxs == expected_idxs)

    # test 2
    state = torch.zeros(1,3,5,5, dtype=torch.bool)
    state[0, 0, 4, 4] = 1 # P1 at (4,4)
    state[0, 1, 2, 3] = 1 # P2 at (2,3)
    state[0, 2, ...] = 7 # 7 moves played

    mask_idxs = two_kings.action_mask(state)[0].nonzero()
    expected_idxs = torch.tensor([[0, 4, 4],
                                  [2, 4, 4]])

    assert torch.all(mask_idxs == expected_idxs)


def test_EnvTwoKings():
    # initialize board: white on c1 ([4,2]), black on c5 ([0,2])
    env = two_kings.EnvTwoKings()

    #############################  White c1 to d1  #############################

    # move white from c1 to d1
    row, col = 4, 2 # c1
    direction = 3 # right
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # white should now be on d1; this is [0,1] (board is now from black's POV)
    assert torch.all(state[1].nonzero() == torch.tensor([[0, 1]])), 'incorrect white pos'

    # black should still be on c5; this is [4,2] (board is now from black's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[4, 2]])), 'incorrect black pos'
    
    # result should be None
    assert result is None
    
    # color should be black
    assert env.color == 'black'

    # move count should be 1
    assert env.move == 1

    #############################  Black c5 to c4  #############################

    # move black from c5 to c4
    row, col = 4, 2 # c5 from POV of black
    direction = 0 # up (from POV of black)
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # white should still be on d1; this is [4,3] (board is now from white's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[4, 3]])), 'incorrect white pos'

    # black should now be on c4; this is [1,2] (board is now from white's POV)
    assert torch.all(state[1].nonzero() == torch.tensor([[1, 2]])), 'incorrect black pos'
    
    # result should be None
    assert result is None

    # color should be white
    assert env.color == 'white'

    # move count should be 2
    assert env.move == 2

    #############################  White d1 to d2  #############################

    # move white from d1 to d2
    row, col = 4, 3 # d1
    direction = 0 # up
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # white should now be on d2; this is [1,1] (board is now from black's POV)
    assert torch.all(state[1].nonzero() == torch.tensor([[1, 1]])), 'incorrect white pos'

    # black should still be on c4; this is [3,2] (board is now from black's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[3, 2]])), 'incorrect black pos'
    
    # result should be None
    assert result is None
    
    # color should be black
    assert env.color == 'black'

    # move count should be 2
    assert env.move == 2

    #############################  Black c4 to c3  #############################

    # move black from c4 to c3
    row, col = 3, 2 # c4 from POV of black
    direction = 0 # up (from POV of black)
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # white should still be on d2; this is [3,3] (board is now from white's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[3, 3]])), 'incorrect white pos'

    # black should now be on c3; this is [2,2] (board is now from white's POV)
    assert torch.all(state[1].nonzero() == torch.tensor([[2, 2]])), 'incorrect black pos'
    
    # result should be None
    assert result is None

    # color should be white
    assert env.color == 'white'

    # move count should be 3
    assert env.move == 3

    #############################  White d2 to c2  #############################

    # move white from d2 to c2
    row, col = 3, 3 # d2
    direction = 2 # left
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # white should now be on c2; this is [1,2] (board is now from black's POV)
    assert torch.all(state[1].nonzero() == torch.tensor([[1, 2]])), 'incorrect white pos'

    # black should still be on c3; this is [2,2] (board is now from black's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[2, 2]])), 'incorrect black pos'
    
    # result should be None
    assert result is None
    
    # color should be black
    assert env.color == 'black'

    # move count should be 3
    assert env.move == 3

    #######################  Black c3 to c2 (black wins)  ######################

    # move black from c3 to c2
    row, col = 2, 2 # c3 from POV of black
    direction = 0 # up (from POV of black)
    action = torch.Tensor([[direction, row, col]]).to(torch.int)
    state, result = env.step(action)

    # because game is over, board doesn't get flipped: still in black's POV

    # white's king got taken (all squares should now be zero)
    assert torch.all(state[1].nonzero() == torch.empty((0,2))), 'incorrect white pos'

    # black should now be on c2; this is [1,2] (board is still in blacks's POV)
    assert torch.all(state[0].nonzero() == torch.tensor([[1, 2]])), 'incorrect black pos'
    
    # result should be 'black'
    assert result == 'black'
