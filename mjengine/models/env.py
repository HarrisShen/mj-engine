import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces.utils import flatten_space

from mjengine.constants import GameStatus
from mjengine.game import Game
from mjengine.models.utils import parse_action
from mjengine.shanten import Shanten

"""
position: 0 - 3, representing the player's position on table
hand: 34 dimensions, representing the player's hand
exposed: 56 dimensions, representing the player's exposed tiles and melds
discards: 35 dimensions, representing the player's discards
For discards and exposed, value 0 is reserved to represent the absence of tiles/melds
"""
MAIN_OBSERVATION_SPACE = Dict({
    "position": Discrete(4),
    "hand": MultiDiscrete([4] * 34),
    "exposed": MultiDiscrete([4] * 34),
    "discards": MultiDiscrete([35] * 33),  # How large can discards be?? Now it's just a mitigation.
})


OPPONENT_OBSERVATION_SPACE = Dict({
    "position": Discrete(4),
    "exposed": MultiDiscrete([4] * 34),
    "discards": MultiDiscrete([35] * 33),
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
Action is encoded as a single integer (0 - 75).
0 - 33: discard
34 - 67: concealed kong
68: win from self
69 - 71: chow
72: pong
73: exposed kong
74: win from chuck
75: pass
"""
MAHJONG_ACTION_SPACE = Dict({
    "discard": Discrete(34),
    "concealed_kong": Discrete(34),
    "win_from_self": MultiBinary(1),    
    "chow": Discrete(3),
    "pong": MultiBinary(1),
    "exposed_kong": MultiBinary(1),
    "win_from_chuck": MultiBinary(1),
    "pass": MultiBinary(1),
})


def step_reward(start: int, finish: int) -> int:
    if start < finish:
        return -max(1, step_reward(finish, start) // 4)
    reward = 0
    for i in range(start, finish, -1):
        reward += 2 ** (6 - i) if i < 7 else 1
    return reward


class MahjongEnv(gym.Env):
    def __init__(
            self,
            game: Game | None = None,
            index_dir: str = "."):
        self.game = game
        if self.game is None:
            self.game = Game(verbose=False)

        self.shanten = Shanten()
        self.shanten.prepare(index_dir)

        self.action_space = flatten_space(MAHJONG_ACTION_SPACE)
        self.observation_space = flatten_space(MAHJONG_OBSERVATION_SPACE)

    def step(self, action) -> tuple[np.ndarray, int, bool, bool, dict]:
        reward = 0
        action_code, tile = parse_action(action)
        if action > 68 and self.game.players[self.game.current_player].discards:
            tile = self.game.players[self.game.current_player].discards[-1]
        old_st = self.shanten(self.game.players[self.game.acting_player].hand)
        try:
            self.game.apply_action(action_code, tile)
        except ValueError:
            reward = -100
            info = {
                "option": self.game.option.to_numpy(),
                "next_player_state": self.game.to_numpy()  # state in next player's perspective
            }
            return self.game.to_numpy(), reward, False, False, info
        player = self.game.players[self.game.acting_player]
        if self.game.status == GameStatus.END:
            if player.won:
                reward = 128
            return self.game.to_numpy(), reward, True, False, {
                "option": None,
                "next_player_state": None,
                "chuck_tile": tile if player.won else None
            }
        reward = step_reward(old_st, self.shanten(player.hand))
        state = self.game.to_numpy()
        self.game.get_option()
        next_player_state = self.game.to_numpy()
        info = {
            "option": self.game.option.to_numpy(),
            "next_player_state": next_player_state
        }
        return state, reward, False, False, info

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        self.game.start_game()
        self.game.get_option()
        return self.game.to_numpy(), {"option": self.game.option.to_numpy()}

    def render(self):
        pass
