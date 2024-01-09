from collections import Counter
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces.utils import flatten_space
import numpy as np

from mjengine.constants import GameStatus, PlayerAction
from mjengine.game import Game
from mjengine.utils import distance_to_ready


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
    "discards": MultiDiscrete([35] * 21),
})


OPPONENT_OBSERVATION_SPACE = Dict({
    "position": Discrete(4),
    "exposed": MultiDiscrete([4] * 34),
    "discards": MultiDiscrete([35] * 21),
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
    "discard": Discrete(34),
    "concealed_kong": Discrete(34),
    "win_from_self": MultiBinary(1),    
    "chow": Discrete(3),
    "pong": MultiBinary(1),
    "exposed_kong": MultiBinary(1),
    "win_from_chuck": MultiBinary(1),
    "pass": MultiBinary(1),
})


def encode_tile(tid: int) -> int:
    """
    Encode tile ID to tile index
    Compress the range of tile IDs to [1, 34]"""
    if tid < 40:
        return (tid // 10 - 1) * 9 + tid % 10
    elif tid < 48:
        return 28 + (tid - 40) // 2
    else:
        return 32 + (tid - 50) // 2
    

def get_state(game: Game, player: int) -> np.ndarray:
    state = game.to_dict(as_player=player)
    encoded_state = np.array([])
    for i in range(4):
        pid = (player + i) % 4
        if i == 0:
            encoded_hand = np.zeros(34, dtype=np.int32)
            counter = Counter(state["players"][player]["hand"])
            for tid, count in counter.items():
                encoded_hand[encode_tile(tid) - 1] += count
        else:
            encoded_hand = np.array([])
        encoded_exposed = np.zeros(34, dtype=np.int32)
        for meld in state["players"][pid]["exposed"]:
            for tid in meld:
                encoded_exposed[encode_tile(tid) - 1] += 1
        encoded_discards = np.zeros(21, dtype=np.int32)
        for i, tid in enumerate(state["players"][pid]["discards"]):
            encoded_discards[i] = encode_tile(tid)
        encoded_state = np.concatenate([
            encoded_state, [pid], 
            encoded_hand, encoded_exposed,
            encoded_discards]).astype(np.int32)
    encoded_state = np.concatenate([
        encoded_state, 
        [state["wall"], state["dealer"], state["current_player"]]
    ]).astype(np.int32)
    return encoded_state


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
    def __init__(self, game: Game):
        self.game = game

        self.action_space = flatten_space(MAHJONG_ACTION_SPACE)
        self.observation_space = flatten_space(MAHJONG_OBSERVATION_SPACE)

    def step(self, action):
        reward = 0
        try:
            action, hand_index = parse_action(action)
            tile = None if hand_index is None else self.player.hand[hand_index]
            self.game.apply_action(action, [self.pid], tile)
        except ValueError:
            reward = -10
            return get_state(self.game), reward, False, False, {}
        if self.game.status == GameStatus.END:
            if self.player.won:
                reward = 100
            elif any([player.is_winning() for player in self.game.players]):
                reward = -10
            return get_state(self.game), reward, True, False, {}
        reward = 7 - distance_to_ready(self.player.hand)
        return get_state(self.game), reward, False, False, {}

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        self.game.start_game()
        return get_state(self.game), {}

    def render(self):
        pass
