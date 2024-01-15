from collections import Counter
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces.utils import flatten_space
import numpy as np

from mjengine.constants import GameStatus, PlayerAction
from mjengine.game import Game
from mjengine.utils import distance_to_ready_old


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
    # TODO: move acquired tiles out from discards
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


def encode_tile(tid: int) -> int:
    """
    Encode tile ID to tile index
    Compress the range of tile IDs to [1, 34]"""
    if tid < 40:
        return (tid // 10 - 1) * 9 + tid % 10
    if tid < 48:
        return 28 + (tid - 40) // 2
    return 32 + (tid - 50) // 2


def decode_tile(code: int) -> int:
    """
    Decode tile code [1, 34] back to tile ID
    """
    if code < 28:
        return 10 * (1 + (code - 1) // 9) + (code - 1) % 9 + 1
    if code < 32:
        return 2 * code - 15
    return 2 * code - 13


def get_state(game: Game, player: int | None = None) -> np.ndarray:
    if player is None:
        player = game.acting_player
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
        encoded_discards = np.zeros(33, dtype=np.int32)
        for j, tid in enumerate(state["players"][pid]["discards"]):
            encoded_discards[j] = encode_tile(tid)
        encoded_state = np.concatenate([
            encoded_state, [pid], 
            encoded_hand, encoded_exposed,
            encoded_discards]).astype(np.int32)
    encoded_state = np.concatenate([
        encoded_state, 
        [state["wall"], state["dealer"], state["current_player"]]
    ]).astype(np.int32)
    return encoded_state


def parse_action(action: int | np.ndarray) -> tuple[PlayerAction | int | None, int | None]:
    if isinstance(action, np.ndarray):
        action = action.flatten(order="C")
        if len(action) != 76:
            raise ValueError("Invalid array for action")
        action = np.argmax(action)
    if not isinstance(action, int):
        raise ValueError("Invalid parameter type for action")
    if action < 0 or action > 75:
        raise ValueError("Invalid action")
    if 0 <= action <= 33:
        return None, decode_tile(action + 1)
    elif 34 <= action <= 67:
        return PlayerAction.KONG, decode_tile(action - 33)
    elif action == 68:
        return PlayerAction.WIN, None
    elif 69 <= action <= 71:
        return PlayerAction.CHOW1 + action - 69, None
    elif action == 72:
        return PlayerAction.PONG, None
    elif action == 73:
        return PlayerAction.KONG, None
    elif action == 74:
        return PlayerAction.WIN, None
    return PlayerAction.PASS, None


def get_option(game: Game) -> np.ndarray:
    option = np.zeros(76, dtype=bool)
    if game.option.discard:
        hand = set(encode_tile(tid) for tid in game.players[game.current_player].hand)
        for i in range(34):
            if i + 1 in hand:  # note the diff between tile encoding and action codes
                option[i] = True
    elif game.option.concealed_kong:
        for tid in game.option.concealed_kong:
            option[encode_tile(tid) + 33] = True
    elif game.option.win_from_self:
        option[68] = True
    elif game.option.chow[0]:
        for i in range(1, 4):
            option[68 + i] = game.option.chow[i]
    elif game.option.pong:
        option[72] = True
    elif game.option.exposed_kong:
        option[73] = True
    elif game.option.win_from_chuck:
        option[74] = True
    # enable pass for non-discard situations
    if not game.option.discard:
        option[75] = True
    return option


class MahjongEnv(gym.Env):
    def __init__(self, game: Game | None = None):
        self.game = game
        if self.game is None:
            self.game = Game(verbose=False)

        self.action_space = flatten_space(MAHJONG_ACTION_SPACE)
        self.observation_space = flatten_space(MAHJONG_OBSERVATION_SPACE)

    def step(self, action):
        reward = 0
        action_code, tile = parse_action(action)
        if 67 < action < 75:
            tile = self.game.players[self.game.current_player].discards[-1]
        old_hand = self.game.players[self.game.acting_player].hand.copy()
        try:
            self.game.apply_action(action_code, tile)
        except ValueError:
            reward = -100
            info = {
                "option": get_option(self.game),
                "next_player_state": get_state(self.game)  # state in next player's perspective
            }
            return get_state(self.game), reward, False, False, info
        player = self.game.players[self.game.acting_player]
        if self.game.status == GameStatus.END:
            if player.won:
                reward = 100
            elif any([player.is_winning() for player in self.game.players]):
                reward = -10
            return get_state(self.game), reward, True, False, {"option": None, "next_player_state": None}
        reward = distance_to_ready_old(old_hand) - distance_to_ready_old(player.hand)
        state = get_state(self.game)
        self.game.get_option()
        next_player_state = get_state(self.game)
        info = {
            "option": get_option(self.game),
            "next_player_state": next_player_state
        }
        return state, reward, False, False, info

    def reset(self, *, seed=None, options=None):
        self.game.reset()
        self.game.start_game()
        self.game.get_option()
        return get_state(self.game), {"option": get_option(self.game)}

    def render(self):
        pass
