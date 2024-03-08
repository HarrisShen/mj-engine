import os
import pickle

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Dict, Discrete, MultiBinary
from gymnasium.spaces.utils import flatten_space

from mjengine.analyzer import Analyzer
from mjengine.constants import GameStatus
from mjengine.game import Game
from mjengine.models.utils import parse_action

"""
position: 0 - 3, representing the player's position on table
hand: 34 dimensions, representing the player's hand
exposed: 56 dimensions, representing the player's exposed tiles and melds
discards: 35 dimensions, representing the player's discards
For discards and exposed, value 0 is reserved to represent the absence of tiles/melds
"""
MAIN_OBSERVATION_SPACE = Dict({
    "position": Discrete(1),
    "hand": Discrete(34),
    "exposed": Discrete(34),
    "discards": Discrete(33),  # How large can discards be?? Now it's just a mitigation.
})


OPPONENT_OBSERVATION_SPACE = Dict({
    "position": Discrete(1),
    "exposed": Discrete(34),
    "discards": Discrete(33),
})


MAHJONG_OBSERVATION_SPACE = Dict({
    "player1": MAIN_OBSERVATION_SPACE,
    "player2": OPPONENT_OBSERVATION_SPACE,
    "player3": OPPONENT_OBSERVATION_SPACE,
    "player4": OPPONENT_OBSERVATION_SPACE,
    "wall": Discrete(1),
    "dealer": Discrete(1),
    "current_player": Discrete(1),
    "acting_player": Discrete(1)
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


def step_reward(shanten_real: int, n_exp: int, n_round: int) -> float:
    n_exp_base = min(136, 36 + 20 * (shanten_real - 1))
    low = -(2 ** (shanten_real - 4))
    coef = (1 - min(1.0, n_exp / n_exp_base)) * np.exp((n_round - 8) / 30)
    late_pen = np.ceil(-np.log(30 - n_round) * 1.22 + 4) / 2  # (n_round // 3) / 3
    return low * coef - late_pen


class MahjongEnv(gym.Env):
    def __init__(
            self,
            game: Game | None = None,
            seed: int | None = 0,
            index_dir: str = "./index/",
            wall_file: str = ""):
        self.game = game
        if self.game is None:
            self.game = Game(verbose=False, wall_file=wall_file)
        self.seed(seed)

        self.analyzer = Analyzer()
        self.analyzer.prepare(index_dir)

        self.action_space = flatten_space(MAHJONG_ACTION_SPACE)
        self.observation_space = flatten_space(MAHJONG_OBSERVATION_SPACE)

    def seed(self, sd: int | None):
        self.game.set_seed(sd)

    def prepare_analyzer(self, index_dir: str = "./index/"):
        self.analyzer = Analyzer()
        self.analyzer.prepare(index_dir)

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = 0
        action_code, tile = parse_action(action)
        if action > 68 and self.game.players[self.game.current_player].discards:
            tile = self.game.players[self.game.current_player].discards[-1]
        acting_player = self.game.acting_player
        player = self.game.players[acting_player]
        try:
            self.game.apply_action(action_code, tile)
        except ValueError:
            print(f"Invalid action {action} noted")
            reward = -100.0
            info = {
                "option": self.game.option.to_numpy(),
                "next_player_state": self.game.to_numpy()  # state in next player's perspective
            }
            return self.game.to_numpy(), reward, False, False, info
        if self.game.status == GameStatus.END:
            if player.won:
                reward = 128.0
            return self.game.to_numpy(), reward, True, False, {
                "option": None,
                "next_player_state": None,
                "chuck_tile": tile if player.won else None,
            }
        new_st, _, new_wait = self.analyzer(player.hand)
        new_n_exp = sum(self.game.tiles_left(acting_player, new_wait))
        n_round = len(self.game.players[acting_player].discards)
        reward = step_reward(new_st, new_n_exp, n_round)
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

    def save(self, out_dir):
        self.analyzer = None
        with open(os.path.join(out_dir, "mahjong_env.pkl"), "wb") as f:
            pickle.dump({"env": self, "game_rng": self.game.r.getstate()}, f)

    @staticmethod
    def restore(dirpath):
        with open(os.path.join(dirpath, "mahjong_env.pkl"), "rb") as f:
            pickled = pickle.load(f)
        env = pickled["env"]
        env.game.r.setstate(pickled["game_rng"])
        env.prepare_analyzer()
        return env
