import os

import numpy as np
import torch

from mjengine.constants import GameStatus, PlayerAction
from mjengine.models.agent import Agent


class Deterministic(Agent):
    def __init__(self, game, strategy: str):
        super().__init__(False)

        from mjengine.player import make_player

        self.make_player = make_player
        self.game = game

        self.strategy = strategy
        self.game.players = [make_player(strategy) for _ in range(4)]

    def take_action(self, state: np.ndarray, option: np.ndarray) -> int:
        game = self.game
        last_discard = None
        if game.status == GameStatus.CHECK:
            last_discard = game.players[game.current_player].discards[-1]
        action, tile = game.players[game.acting_player].decide(
            game.option,
            last_discard,
            game.to_dict(game.acting_player))
        if action is None:
            return tile
        if action == PlayerAction.PASS:
            return 75
        if action == PlayerAction.WIN:
            return 68 if tile is None else 74
        if action == PlayerAction.KONG:
            return 34 + tile if game.option.concealed_kong else 73
        if action == PlayerAction.PONG:
            return 72
        # CHOW
        return 68 + int(action)

    def update(self, **kwargs) -> None:
        pass

    def save(self, model_dir, checkpoint=None) -> str:
        if checkpoint is not None:
            raise ValueError("Cannot set checkpoint for deterministic agent")
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir

    @staticmethod
    def restore(model_dir: str, device: torch.device, train=False):
        raise RuntimeError("Deterministic agent cannot be restored from files")
