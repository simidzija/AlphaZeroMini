from typing import Protocol, Optional
import torch

class EnvProtocol(Protocol):
    in_features: int
    board_size: int
    state: torch.BoolTensor
    move_limit: int

    @property
    def color(self) -> str:
        ...
    
    @property
    def move(self) -> int:
        ...

    def new_env(self, state: Optional[torch.Tensor]) -> 'EnvProtocol':
        ...

    def clone(self) -> 'EnvProtocol':
        ...

    def get_square(self, action: torch.Tensor) -> str:
        ...
        
    def step(self, action: torch.IntTensor, update_state: bool, print_move: bool) -> tuple:
        ...

    def get_pos_dict(self, perspective: str) -> dict:
        ...

    