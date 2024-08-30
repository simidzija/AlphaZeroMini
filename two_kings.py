import torch
import network
import pygame
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def action_mask(state: torch.Tensor) -> torch.BoolTensor:
    """Mask of allowed actions.

    :param state: bx4x5x5 tensor. Each batch b has 4 5x5 planes: P1 pos, P2 pos, color, move number
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
    if not state[0, r_start, c_start]:
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
    def __init__(self, color=None):
        self.in_features = 4
        self.board_size = 5
        self.move_limit = 10
        self.state = torch.zeros(self.in_features, self.board_size, 
                                 self.board_size)
        
        # Initialize P1 and P2 planes
        self.state[0, 4, 2] = 1 # P1 at c1
        self.state[1, 0, 2] = 1 # P2 at c5

        # Intitialize color plane
        if color is None or color == 'white':
            self.state[2] = -1
        elif color == 'black':
            self.state[2] = 1
        else:
            raise ValueError(f"color must be 'black', 'white' or None but got {color}")

        # Initialize move plane
        self.state[3] = 1

    @property
    def color(self):
        if self.state[2, 0, 0] == -1:
            return 'white'
        elif self.state[2, 0, 0] == 1:
            return 'black'
        else:
            raise RuntimeError(f'self.state[2] must have all elements equal to 0 or 1, but got {self.state[2]}')

    @property
    def move(self):
        return int(self.state[3, 0, 0].item())


    def step(self, action: torch.IntTensor) -> tuple:
        """Update env based on action.
        
        :param action: int tensor [[dir, row, col]] (batch size of 1)
        :return: tuple (state, result) where:
            state: tensor of shape (1,4,5,5)
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
                self.state[2] = -1 # change color to white
                self.state[3] += 1 # increase move count
            else:
                self.state[2] = 1 # change color to black
            return self.state, None

    def get_pos_dict(self, perspective: str) -> dict:
        """Dict of positions from given perspective ('white' or 'black')."""

        assert perspective == 'white' or perspective == 'black', f"perspective must be 'white' or 'black' but got {perspective}."

        P1_K = tuple(self.state[0].nonzero().squeeze().tolist())
        P2_K = tuple(self.state[1].nonzero().squeeze().tolist())

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

def play():
    # Initialize env and NN
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
    net = network.Network(**game_params, **architecture_params)

    # Initialize pygame
    pygame.init()

    # Board and square sizes
    WIDTH, HEIGHT = 1100, 1100
    BOARD_SIZE = 5
    SQUARE_SIZE = 200
    LABEL_SIZE = 50

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
        window.blit(title, (WIDTH/2 - title.get_width()/2, 0.2 * HEIGHT))

        color_prompt = font.render("CHOOSE YOUR COLOUR:", True, WHITE)
        window.blit(color_prompt, (WIDTH/2 - title.get_width()/2, 0.3 * HEIGHT))

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
        if env.state[0, r, c]:
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
                if ai_color == 'white' and env.move == 1:
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
                            print('Invalid move')
                        else:
                            print(f'Piece moved to pos {piece_pos_end}')
                            piece_selected = False
                            action = get_action(piece_pos_start, piece_pos_end)
                            _, result = env.step(action)
                            coords_dict = get_coords_dict(
                                env.get_pos_dict(player_color))
                            if result is None:
                                player_move = False
                            pygame.event.post(fake_event)
                    elif not piece_selected:
                        if piece_pos_start is None:
                            print('Invalid piece selection')
                        else:
                            piece_selected = True
                            print(f'Piece selected at pos {piece_pos_start}')

            elif not player_move:
                action = net.greedy_sample(env.state)
                _, result = env.step(action)
                coords_dict = get_coords_dict(env.get_pos_dict(player_color))
                if result is None:
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
    play()

