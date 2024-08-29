from typing import Protocol
import torch

class EnvProtocol(Protocol):
    in_features: int
    board_size: int
    state: torch.BoolTensor
    color: str
    move: int
    move_limit: int

    def step(self, action: torch.IntTensor) -> tuple:
        ...

    def get_pos_dict(self, perspective: tuple) -> dict:
        ...

    