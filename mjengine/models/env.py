import gymnasium as gym
from gymnasium.spaces import Space, Dict, Discrete, MultiDiscrete, MultiBinary
import numpy as np

from mjengine.constants import GameStatus
from mjengine.game import Game
from mjengine.utils import distance_to_ready


MAIN_OBSERVATION_SPACE = Dict({
    "position": Discrete(4),
    "hand": MultiDiscrete([4] * 34),
    "discards": MultiDiscrete([34] * 21),
    "exposed": MultiDiscrete([55] * 4),
})


OPPONENT_OBSERVATION_SPACE = Dict({
    "position": Discrete(4),
    "discards": MultiDiscrete([34] * 21),
    "exposed": MultiDiscrete([55] * 4),
})


MAHJONG_OBSERVATION_SPACE = Dict({
    "player1": MAIN_OBSERVATION_SPACE,
    "player2": OPPONENT_OBSERVATION_SPACE,
    "player3": OPPONENT_OBSERVATION_SPACE,
    "player4": OPPONENT_OBSERVATION_SPACE,
    "wall": Discrete(136),
    "dealer": Discrete(4),
    "current_player": Discrete(4),
})


MAHJONG_ACTION_SPACE = Dict({
    "discard": Discrete(14),
    "concealed_kong": Discrete(14),
    "win_from_self": MultiBinary(1),    
    "chow": Discrete(3),
    "pong": MultiBinary(1),
    "exposed_kong": MultiBinary(1),
    "win_from_chuck": MultiBinary(1),
})


#TODO
###############################################
def get_state(game: Game) -> np.ndarray:
    pass


def action_to_dict(action: np.ndarray) -> dict:
    pass


def is_valid_action(action: dict) -> bool:
    pass
###############################################

class MahjongEnv(gym.Env):
    def __init__(self):
        self.game = Game()
        self.game.deal()

        self.action_space = Space()
        self.observation_space = MAHJONG_OBSERVATION_SPACE

    def step(self, action):
        reward = 0
        action = action_to_dict(action)
        if not is_valid_action(action):
            reward = -10
            return get_state(self.game), reward, False, False, {}
        active_player = self.game.players[self.game.active_player]
        try:
            self.game.next_action(action)
        except ValueError:
            reward = -10
            return get_state(self.game), reward, False, False, {}
        if self.game.status == GameStatus.END:
            if self.game.players[self.game.active_player].won:
                reward = 10
            elif any([player.is_winning() for player in self.game.players]):
                reward = -10
            return get_state(self.game), reward, True, False, {}
        reward = 7 - distance_to_ready(active_player.hand)
        return get_state(self.game), reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        self.game.deal()
        return get_state(self.game), {}

    def render(self):
        pass
