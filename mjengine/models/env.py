import gymnasium as gym
from gymnasium.spaces import Space, Dict, Discrete, MultiDiscrete, MultiBinary
import numpy as np

from mjengine.constants import GameStatus, PlayerAction
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


"""
Action is encoded as a single integer (0 - 35).
0 - 13: discard
14 - 27: concealed kong
28: win from self
29 - 31: chow
32: pong
33: exposed kong
34: win from chuck
35: pass
"""
MAHJONG_ACTION_SPACE = Dict({
    "discard": Discrete(14),
    "concealed_kong": Discrete(14),
    "win_from_self": MultiBinary(1),    
    "chow": Discrete(3),
    "pong": MultiBinary(1),
    "exposed_kong": MultiBinary(1),
    "win_from_chuck": MultiBinary(1),
    "pass": MultiBinary(1),
})


#TODO
###############################################
def get_state(game: Game) -> np.ndarray:
    pass


def parse_action(action: int | np.ndarray) -> tuple[PlayerAction, int]:
    if isinstance(action, np.ndarray):
        action = action.flatten(order="C")
        if len(action) != 36:
            raise ValueError("Invalid array for action")
        action = np.argmax(action)
    if not isinstance(action, int):
        raise ValueError("Invalid parameter type for action")
    if action < 0 or action > 35:
        raise ValueError("Invalid action")
    if 0 <= action <= 13:
        return None, action
    elif 14 <= action <= 27:
        return PlayerAction.KONG, action - 14
    elif action == 28:
        return PlayerAction.WIN, None
    elif 29 <= action <= 31:
        return PlayerAction.CHOW1 + action - 29, None
    elif action == 32:
        return PlayerAction.PONG, None
    elif action == 33:
        return PlayerAction.KONG, None
    elif action == 34:
        return PlayerAction.WIN, None
    return PlayerAction.PASS, None


class MahjongEnv(gym.Env):
    def __init__(self):
        self.game = Game()
        self.game.deal()

        self.action_space = Space()
        self.observation_space = MAHJONG_OBSERVATION_SPACE

    def step(self, action):
        reward = 0
        try:
            action, mode = parse_action(action)
            active_player = self.game.players[self.game.active_player]
            self.game.next_action(action=action, tile=tile)
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
