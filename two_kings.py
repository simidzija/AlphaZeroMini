"""
Two Kings.

This module implements the Two Kings game, a simplified version of chess played on a 5x5 board between only two kings. The rules of the game are as follows:

The kings start on opposite sides of the board and they can only move horizontally and vertically (not diagonally). White moves first, and the goal of the game is to take the opponents king. If no one succeeds to do so within 10 moves, the game is declared a draw.

Because diagonal moves are not allowed, the game is such that it is impossible for white to win, and optimal play results in a black win in 5 moves.

This module implements various methods / classes. The most high level ones are:
    - EnvTwoKings: class for the Two Kings game environment
    - play: method to play the Two Kings game in a GUI, against a neural network
    
Running this file as a script calls play(), allowing the user to play Two Kings against a neural network.
"""

import torch
from network import Network
from mcts import Tree, mcts

import pygame
import os
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def action_mask(state: torch.Tensor) -> torch.BoolTensor:
    """Mask of allowed actions.

    :param state: Each batch has 4 board planes: P1 pos, P2 pos, color, move number
    :return mask: Each batch has 4 board planes: move up, down, left, right
    """
    # TODO: Figure out a more efficient / "torchy" implementation

    board_size = 5

    num_batches = state.size(0)
    mask = torch.zeros(num_batches, 4, board_size, board_size, dtype=torch.bool)

    for b in range(num_batches):
        row, col = state[b, 0].nonzero(as_tuple=True)

        if row > 0:
            mask[b, 0, row, col] = 1
        if row < board_size - 1:
            mask[b, 1, row, col] = 1
        if col > 0:
            mask[b, 2, row, col] = 1
        if col < board_size - 1:
            mask[b, 3, row, col] = 1
    
    return mask

def get_move_type(pos_start: tuple, pos_end: tuple) -> str:
    """Return 0 (up), 1 (down), 2 (left), 3 (right) or None (invalid move)."""
    # check if positions are valid squares
    r_start, c_start = pos_start
    r_end, c_end = pos_end
    if min(r_start, c_start, r_end, c_end) < 0:
        return False
    if max(r_start, c_start, r_end, c_end) >= 5:
        return False 
    
    if pos_end == (r_start - 1, c_start):
        return 0
    elif pos_end == (r_start + 1, c_start):
        return 1 
    elif pos_end == (r_start, c_start - 1):
        return 2
    elif pos_end == (r_start, c_start + 1):
        return 3
    else:
        return None

def allowed_action(state: torch.Tensor, pos_start, pos_end) -> bool:
    """Return True iff it's legal to move P1 piece from pos_start to pos_end."""
    move_type = get_move_type(pos_start, pos_end)
    if move_type is None:
        return False

    r_start, c_start = pos_start
    if not state[0, 0, r_start, c_start]:
        return False
    else:
        return True
    
def get_action(pos_start: tuple, pos_end: tuple) -> torch.IntTensor:
    """Return action corresponding to picking up piece on pos_start and 
    putting down piece on pos_end. If invalid move, return None.
    
    :param pos_start: tuple (r_start, c_start)
    :param pos_end: tuple (r_end, c_end)
    :return: tensor 
    """
    move_type = get_move_type(pos_start, pos_end)
    if move_type is None:
        return None

    row, col = pos_start
    return torch.tensor([[move_type, row, col]], dtype=torch.int)


class EnvTwoKings:
    """Environment for Two Kings game."""
    def __init__(self, state=None):
        self.in_features = 4
        self.board_size = 5
        self.move_limit = 10

        if isinstance(state, torch.Tensor):
            self.state = state
        elif state is None:
            self.state = torch.zeros(1, self.in_features, self.board_size, 
                                     self.board_size)
            self.state[0, 0, 4, 2] = 1 # P1 at c1
            self.state[0, 1, 0, 2] = 1 # P2 at c5
            self.state[0, 2] = -1 # white to move
            self.state[0, 3] = 1 # move 1
        else:
            raise ValueError(f'state must be tensor or None, but got type {state}')

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

    def new_env(self, state=None):
        return type(self)(state=state)

    def clone(self):
        state = self.state.clone()
        return self.new_env(state=state)
    
    def get_move(self, action: torch.Tensor, color: Optional[str]=None) -> str:
        """Return move when player of given color takes given action.
        
        :param action: index tensor [[dir, row, col]]
        :param color: 'black', 'white', or None
        :return move: move in chess notation
        """
        color = self.color if color is None else color

        if color == 'white':
            ROWS = list('87654321')[-self.board_size:]
            COLS = list('abcdefgh')[:self.board_size]
        elif color == 'black':
            ROWS = list('12345678')[:self.board_size]
            COLS = list('hgfedcba')[-self.board_size:]
        else:
            raise RuntimeError(f'color must be "black" or "white" but got {color}')

        idx_d, idx_r, idx_c = action.squeeze().tolist()

        if idx_d == 0: # up
            row, col = ROWS[idx_r - 1], COLS[idx_c]
        elif idx_d == 1: # down
            row, col = ROWS[idx_r + 1], COLS[idx_c]
        elif idx_d == 2: # left
            row, col = ROWS[idx_r], COLS[idx_c - 1]
        elif idx_d == 3: # right
            row, col = ROWS[idx_r], COLS[idx_c + 1]
        else:
            raise RuntimeError(f'idx_d must be 0, 1, 2, or 3 but got {idx_d}')
        
        move = f'K{col}{row}'

        return move


    def step(self, action: torch.IntTensor, update_state=True, print_move=False) -> tuple:
        """Update env based on action.
        
        :param action: int tensor [[dir, row, col]] (batch size of 1)
        :return: tuple (state, result) where:
            state: tensor of shape (1,4,board_size,board_size)
            result: "white", "black", "draw" or None (if game not over)
        """
        assert action.shape == torch.Size([1,3]), f'action.shape should be [1,3] but got {action.shape}'
        assert action.dtype == torch.int, f'action.dtype should be torch.int but got {action.dtype}'

        if print_move:
            move = self.get_move(action)
            if self.color == 'white':
                print(f'{self.move_num}.{move}', end=' ')
            else:
                print(f'{move}', end=' ')

        if update_state:
            new_state = self.state
        else:
            new_state = self.state.clone()

        # move P1
        P1 = new_state[0, 0]
        P2 = new_state[0, 1]

        direction, row, col = action.squeeze().clone()
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
            result = self.color
            if print_move:
                print(f'# {result} wins')
        # check if move limit reached
        elif self.color == 'black' and self.move_num == self.move_limit:
            result = 'draw'
            if print_move:
                print('# draw')
        else:
            result = None

        # rotate board, change color, increase move count
        new_state[0, 0], new_state[0, 1] = new_state[0, 1].flip(0,1), new_state[0, 0].flip(0,1)
        if self.color == 'black':
            new_state[0, 2] = -1 # change color to white
            new_state[0, 3] += 1 # increase move count
        else:
            new_state[0, 2] = 1 # change color to black

        return new_state.clone(), result

    def get_pos_dict(self, perspective: str) -> dict:
        """Dict of positions from given perspective ('white' or 'black')."""

        assert perspective == 'white' or perspective == 'black', f"perspective must be 'white' or 'black' but got {perspective}."

        P1_K = tuple(self.state[0, 0].nonzero().squeeze().tolist())
        P2_K = tuple(self.state[0, 1].nonzero().squeeze().tolist())

        pos_dict = {}

        def rotate(coords):
            if len(coords) == 0: # if piece got taken
                return None
            x, y = coords
            return self.board_size - 1 - x, self.board_size - 1 - y

        if perspective == 'white':
            if self.color == 'white':
                pos_dict['white'] = P1_K if P1_K else None
                pos_dict['black'] = P2_K if P2_K else None
            elif self.color == 'black':
                pos_dict['white'] = rotate(P2_K)
                pos_dict['black'] = rotate(P1_K)
        elif perspective == 'black':
            if self.color == 'white':
                pos_dict['white'] = rotate(P1_K)
                pos_dict['black'] = rotate(P2_K)
            elif self.color == 'black':
                pos_dict['white'] = P2_K if P2_K else None
                pos_dict['black'] = P1_K if P1_K else None

        return pos_dict


def play(net: Network, n_simulations: int, print_move: bool=True):
  
    pygame.init()

    # Board and square sizes
    BOARD_SIZE = 5
    SQUARE_SIZE = 200
    LABEL_SIZE = 50
    WIDTH = SQUARE_SIZE * BOARD_SIZE + 2 * LABEL_SIZE
    HEIGHT = SQUARE_SIZE * BOARD_SIZE + 2 * LABEL_SIZE

    # Colours
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BROWN = (139, 69, 19)
    BEIGE = (245, 245, 220)
    DARK_BROWN = (82, 46, 21)

    # Font
    font = pygame.font.Font(None, 36)
    font_large = pygame.font.Font(None, 72)

    # Set up display
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(("Two Kings"))

    # Pieces
    white_K = pygame.image.load(os.path.join(CURRENT_DIR, 'images/white_K.png'))
    black_K = pygame.image.load(os.path.join(CURRENT_DIR, 'images/black_K.png'))
    white_K = pygame.transform.scale(white_K, (SQUARE_SIZE, SQUARE_SIZE))
    black_K = pygame.transform.scale(black_K, (SQUARE_SIZE, SQUARE_SIZE))

    # Drawing functions

    def make_square(row: int, col: int) -> tuple:
        square = (col * SQUARE_SIZE + LABEL_SIZE, 
                  row * SQUARE_SIZE + LABEL_SIZE, 
                  SQUARE_SIZE, 
                  SQUARE_SIZE)
        return square


    def draw_button(image: pygame.Surface, x, y, w, h, color):
        pygame.draw.rect(window, color, (x, y, w, h))

        w_im = image.get_width()
        h_im = image.get_height()

        window.blit(image, (x + (w - w_im) / 2, y + (h - h_im) / 2))

    def selection_screen():
        window.fill(DARK_BROWN)

        title = font_large.render("TWO KINGS", True, WHITE)
        window.blit(title, (WIDTH/2 - title.get_width()/2, 0.1 * HEIGHT))

        instructions1 = font.render("GAME RULES:", True, WHITE)
        instructions2 = font.render("  - Take your opponent's king", True, WHITE)
        instructions3 = font.render("  - You may only move left, right, up, down", True, WHITE)
        instructions4 = font.render("  - After 10 moves the game is a draw", True, WHITE)
        instructions_width = WIDTH/2 - instructions3.get_width()/2

        # window.blit(instructions1, (instructions_width, 0.2 * HEIGHT))
        window.blit(instructions2, (instructions_width, 0.2 * HEIGHT))
        window.blit(instructions3, (instructions_width, 0.23 * HEIGHT))
        window.blit(instructions4, (instructions_width, 0.26 * HEIGHT))

        color_prompt = font.render("CHOOSE YOUR COLOUR:", True, WHITE)
        window.blit(color_prompt, (WIDTH/2 - color_prompt.get_width()/2, 0.35 * HEIGHT))

        x_white = 0.5 * WIDTH - SQUARE_SIZE
        y_white = 0.5 * HEIGHT - SQUARE_SIZE / 2
        x_black = 0.5 * WIDTH
        y_black = 0.5 * HEIGHT - SQUARE_SIZE / 2
        w, h = SQUARE_SIZE, SQUARE_SIZE

        draw_button(white_K, x_white, y_white, w, h, BEIGE)
        draw_button(black_K, x_black, y_black, w, h, BROWN)

        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif left_button_down(event):
                    x, y = pygame.mouse.get_pos()
                    if x_white < x < x_white + w and y_white < y < y_white + h:
                        color = 'white'
                    elif x_black < x < x_black + w and y_black < y < y_black +h:
                        color = 'black'
                elif left_button_up(event):
                    return color

    def result_screen(player_color, ai_color, coords_dict, result) -> bool:
        """Print result in top margin, offer option for new game or quit.
        
        :param player_color: tuple 'white' or 'black'
        :param ai_color: tuple 'white' or 'black'
        :param coords_dict: dict {'white': (x1,y1), 'black': (x2, y2)}
        :param result: tuple 'white', 'black' or 'draw'
        :return new_game: True iff user wants to play a new game
        """
        window.fill(DARK_BROWN)
        draw_board()
        draw_labels(player_color)
        draw_pieces(coords_dict)

        # Result
        if result == player_color:
            result_text = 'GAME OVER: YOU WIN!'
        elif result == ai_color:
            result_text = 'GAME OVER: AI WINS!'
        elif result == 'draw':
            result_text = "GAME OVER: IT'S A DRAW!"
        result_text = font.render(result_text, True, WHITE)
        window.blit(result_text, (0.3 * WIDTH - result_text.get_width()/2, 
                                  LABEL_SIZE/2 - result_text.get_height()/2))

        # New game and quit buttons
        new_text = font.render('NEW GAME', True, BLACK)
        quit_text = font.render('QUIT', True, BLACK)

        x_new = 0.6 * WIDTH
        y_new = 0.5 * LABEL_SIZE - 0.6 * new_text.get_height()
        w = 1.3 * new_text.get_width()
        h = 1.2 * new_text.get_height()
        draw_button(new_text, x_new, y_new, w, h, BROWN)

        x_quit = x_new + 1.5 * new_text.get_width()
        y_quit = y_new
        draw_button(quit_text, x_quit, y_quit, w, h, BROWN)
        
        pygame.display.flip()

        clicked = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                elif left_button_down(event):
                    x, y = pygame.mouse.get_pos()
                    if x_new < x < x_new + w and y_new < y < y_new + h:
                        clicked = True
                        new_game = True
                    elif x_quit < x < x_quit + w and y_quit < y < y_quit + h:
                        clicked = True
                        new_game = False
                elif left_button_up(event) and clicked:
                    return new_game

    def draw_board():
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = BEIGE if (row + col) % 2 == 0 else BROWN
                rect = make_square(row, col)
                pygame.draw.rect(window, color, rect)

    def draw_labels(player_color):
        # Column labels
        for col in range(BOARD_SIZE):
            if player_color == 'white':
                text = chr(97 + col)
            else:
                text = chr(97 + BOARD_SIZE - col - 1)
            label = font.render(text, True, BEIGE)
            x = (col + 0.5) * SQUARE_SIZE + LABEL_SIZE - label.get_width() / 2
            y = HEIGHT - LABEL_SIZE / 2 - label.get_height() / 2
            window.blit(label, (x,y))

        # Row labels
        for row in range(BOARD_SIZE):
            if player_color == 'white':
                text = str(BOARD_SIZE - row)
            else:
                text = str(row + 1)
            label = font.render(text, True, BEIGE)
            x = LABEL_SIZE / 2 - label.get_width() / 2
            y = (row + 0.5) * SQUARE_SIZE + LABEL_SIZE - label.get_height() / 2
            window.blit(label, (x,y))

    def draw_pieces(coords: dict):
        if coords['white']:
            x_white, y_white = coords['white']
            window.blit(white_K, (x_white, y_white))
        if coords['black']:
            x_black, y_black = coords['black']
            window.blit(black_K, (x_black, y_black))

    def draw_highlight(pos: tuple):
        row, col = pos
        color = BLACK
        rect = make_square(row, col)
        pygame.draw.rect(window, color, rect, width=SQUARE_SIZE // 20)

    # Non-drawing functions

    def get_coords_dict(pos_dict: dict) -> dict:
        coords_dict = {}
        for piece, pos in pos_dict.items():
            if pos is None:
                coords = None
            else:
                r, c = pos
                coords = (c * SQUARE_SIZE + LABEL_SIZE, 
                        r * SQUARE_SIZE + LABEL_SIZE)
            coords_dict[piece] = coords
        return coords_dict

    def left_button_down(event: pygame.event.Event):
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
    
    def left_button_up(event: pygame.event.Event):
        return event.type == pygame.MOUSEBUTTONUP and event.button == 1

    def to_pos(coords: tuple):
        """Return square (r, c) or None if coords isn't within a square."""
        x, y = coords
        if x < LABEL_SIZE or x > WIDTH - LABEL_SIZE:
            return None
        elif y < LABEL_SIZE or y > WIDTH - LABEL_SIZE:
            return None
        else:
            r = (y - LABEL_SIZE) // SQUARE_SIZE 
            c = (x - LABEL_SIZE) // SQUARE_SIZE 
            return r, c

    def select_piece(coords: tuple):
        """Return piece position (r, c) or None if invalid selection."""
        pos = to_pos(coords)
        if pos is None:
            return None
        
        r, c = pos
        if env.state[0, 0, r, c]:
            return r, c
        
        return None

    def drop_piece(pos_start, coords: tuple):
        """Return drop position (r, c) or None if invalid selection."""
        pos_end = to_pos(coords)
        if pos_end is None:
            return None
        elif allowed_action(env.state, pos_start, pos_end):
            return pos_end
        else: 
            return None
        
    # Fake event
    fake_event = pygame.event.Event(pygame.USEREVENT + 1)

    #### Game ####

    running = True
    new_game = True

    # Game loop
    while running:
        for event in pygame.event.get():
            if new_game:
                new_game = False
                env = EnvTwoKings()

                # Intro screen: select colors
                player_color = selection_screen()
                if player_color is None:
                    pygame.quit()
                    return
                ai_color = 'white' if player_color == 'black' else 'black'

                # Initialize variables
                player_move = player_color == 'white'
                piece_selected = False
                coords_dict = get_coords_dict(env.get_pos_dict(player_color))
                result = None

                # Draw initial position
                window.fill(DARK_BROWN)
                draw_board()
                draw_labels(player_color)
                draw_pieces(coords_dict)
                pygame.display.flip()

                # wait a bit on first move if ai is white
                if ai_color == 'white' and env.move_num == 1:
                    pygame.time.wait(1000)

            if player_move:
                if event.type == pygame.QUIT:
                    running = False

                elif left_button_down(event):
                    if piece_selected:
                        piece_pos_end = drop_piece(piece_pos_start, 
                                                   pygame.mouse.get_pos())
                    elif not piece_selected:
                        piece_pos_start = select_piece(pygame.mouse.get_pos())

                elif left_button_up(event):
                    if piece_selected:
                        if piece_pos_end is None:
                            # print('Invalid move')
                            pass
                        else:
                            # print(f'Piece moved to pos {piece_pos_end}')
                            piece_selected = False
                            action = get_action(piece_pos_start, piece_pos_end)
                            _, result = env.step(action, print_move=print_move)
                            coords_dict = get_coords_dict(
                                env.get_pos_dict(player_color))
                            player_move = False
                            pygame.event.post(fake_event)
                    elif not piece_selected:
                        if piece_pos_start is None:
                            # print('Invalid piece selection')
                            pass
                        else:
                            piece_selected = True
                            # print(f'Piece selected at pos {piece_pos_start}')

            elif not player_move:
                # get action using MCTS
                tree = Tree(env=env, net=net, c_puct=0.1, temp=1.0, alpha_dir=1.0, eps_dir=0.0)
                action, _ = mcts(tree, n_simulations)

                # perform action in env
                _, result = env.step(action, print_move=True)

                coords_dict = get_coords_dict(env.get_pos_dict(player_color))
                player_move = True
                
            # Draw
            window.fill(DARK_BROWN)
            draw_board()
            if piece_selected:
                draw_highlight(piece_pos_start)
            draw_labels(player_color)
            draw_pieces(coords_dict)
            pygame.display.flip()

            # Result screen
            if result:
                if result_screen(player_color, ai_color, coords_dict, result):
                    new_game = True
                    pygame.event.post(fake_event)
                else:
                    running = False

    # Quit pygame
    pygame.quit()


if __name__ == '__main__':
    ##############   New NN   ##############

    game_params = {
        'num_in_channels': 4, 
        'board_size': 5,
        'num_out_channels': 4,
        'action_mask': action_mask
    }
    architecture_params = {
        'num_filters': 8,
        'kernel_size': 3,
        'num_res_blocks': 6,
        'num_policy_filters': 2,
        'value_hidden_layer_size': 64,
    }
    net = Network(**game_params, **architecture_params)

    ##############   Trained NN   ##############

    # filename = os.path.join('checkpoints', 'batch_50.pth')
    # net = torch.load(filename)

    play(net, n_simulations=100)
