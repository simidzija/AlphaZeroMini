## Two Kings

A black king and a white king are initialized on opposite ends of a 5 by 5 board. They can only move forwards, backwards, left and right (not diagonally), and they're allowed to get next to and take each other. The player who takes his opponents king wins the game. If no one wins after 10 moves the game is a draw. Perfect play results in a black win in 5 moves.

### Input features 
4 planes of size $5\times 5$. The planes correspond to:
1) P1 position: one-hot 
2) P2 position: one-hot
3) Color: constant (-1 for white, 1 for black)
3) Move count: constant

### Action features 
4 planes of size $5\times 5 = 100$ possible actions. The planes correspond to:
1) Move up
2) Move down
3) Move left
4) Move right
The value of a square in plane i indicates the probability (or logit) for the piece in that square to move in direction i. 
Note that with this representation most moves are illegal.

### Legal moves
Given input features, we can determine the legal moves in the following way:
1) Look at P1 plane to determine position of king
3) Only this location is allowed in the action planes
4) The location also determines which of four action planes are allowed
In total only 2,3, or 4 out of the 100 possible actions are legal; the rest are illegal.

### TODO
- Modify MCTS so that when the agent plays a move the corresponding branch of the tree is kept, rather than starting an entirely new tree (see methods of AlphaGo paper)
