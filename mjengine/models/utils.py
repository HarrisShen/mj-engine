import bz2
import os
import pickle
from collections import deque
import random

import numpy as np

from mjengine.analyzer import Analyzer
from mjengine.constants import PlayerAction


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)
        # self.r = random.Random()

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def __iter__(self):
        yield from self.buffer

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def last_n(self, n):
        return [self.buffer[-(n - i)] for i in range(n)]

    def save(self, model_dir, compression=None):
        filepath = os.path.join(model_dir, "replay_buffer.pkl")
        if compression is None:
            with open(filepath, "wb") as f:
                pickle.dump(self, f)
            return
        if compression == "bz2":
            with bz2.open(filepath + ".bz2", "wb", compresslevel=9) as f:
                data = pickle.dumps(self)
                f.write(data)
            return
        raise ValueError(f"Unsupported compression method: {compression}")

    @staticmethod
    def restore(model_dir):
        filepath = os.path.join(model_dir, "replay_buffer.pkl")
        if os.path.isfile(filepath):
            with open(filepath, "rb") as f:
                rb = pickle.load(f)
            return rb
        if os.path.isfile(filepath + ".bz2"):
            with bz2.open(filepath + ".bz2", "rb", compresslevel=9) as f:
                rb = pickle.load(f)
            return rb
        raise FileNotFoundError(f"File of replay buffer not found in {model_dir}")


def game_numpy_to_dict(state: np.ndarray) -> dict:
    players = [{
        "hand": [0 for _ in range(34)],
        "discards": [],
        "exposed": [[]]
    } for _1 in range(4)]
    players[state[0]]["hand"] = state[1: 35].tolist()
    for i in range(4):
        for j in range(34):
            players[(state[0] + i) % 4]["exposed"][0] += [j] * state[35 + j + 69 * i]
            players[(state[0] + i) % 4]["discards"] += [j] * state[69 + j + 69 * i]
        # for j in range(33):
        #     for k in range(34):
        #         if state[35 + 34 * 34 * i + 34 * j + k] == 0:
        #             continue
        #         players[(state[0] + i) % 4]["discards"].append(k)
        # for t in state[69 + 68 * i: 102 + 68 * i]:
        #     if t == 0:
        #         break
        #     players[(state[0] + i) % 4]["discards"].append(t - 1)
    return {
        "wall": state[-4],
        "dealer": state[-3],
        "current_player": state[-2],
        "acting_player": state[-1],
        "players": players
    }


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
        return None, action
    elif 34 <= action <= 67:
        return PlayerAction.KONG, action - 34
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


def encode_action(action: PlayerAction | None, tile: int | None, donor: int | None) -> int:
    if tile is not None and (tile < 0 or tile > 33):
        raise ValueError(f"Invalid tile: {tile}")
    if action is None:
        if tile is None:
            raise ValueError(f"No tile given for discard")
        return tile
    if action == PlayerAction.KONG:
        if donor is None:
            if tile is None:
                raise ValueError(f"No tile given for self kong")
            return tile + 34
        return 73
    if action == PlayerAction.WIN:
        if donor is None:
            return 68
        return 74
    if action == PlayerAction.PONG:
        return 72
    if action > PlayerAction.PASS:
        return action - PlayerAction.CHOW1 + 69
    return 75  # pass


def find_last_discard(state: np.ndarray) -> int:
    tile, n_discards = -1, 0
    for i in range(4):
        for j in range(33):
            if state[69 + i * 68 + j] == 0:
                if n_discards <= j:
                    tile = state[69 + i * 68 + j] - 1
                    n_discards = j
                break
    return tile


LATEST_ENCODING_VERSION = "0.2.0a"


def game_dict_to_numpy(
        state: dict,
        player: int | None = None,
        analyzer: Analyzer | None = None,
        history: bool = False,
        version: str = "latest") -> np.ndarray:
    if version == "latest":
        version = LATEST_ENCODING_VERSION
    # the state dict of game must be masked for opponents
    if player is None:
        for i in range(4):
            if sum(state["players"][i]["hand"]) > 0:
                player = i
                break
        else:
            raise ValueError("Game dict is not masked, please specify 'as_player' in 'Game.to_dict()'")
    encoded_state = np.array([])
    seen_tiles_cnt = np.zeros(34, dtype=np.int32)
    for i in range(4):
        pid = (player + i) % 4
        if i == 0:
            hand = np.array(state["players"][player]["hand"], dtype=np.int32)
            seen_tiles_cnt = np.add(seen_tiles_cnt, hand)
        else:
            hand = np.array([])
        exposed = np.zeros(34, dtype=np.int32)
        for meld in state["players"][pid]["exposed"]:
            for tid in meld:
                exposed[tid] += 1
        seen_tiles_cnt = np.add(seen_tiles_cnt, exposed)
        discards_cnt = np.zeros(34, dtype=np.int32)
        for j, tid in enumerate(state["players"][pid]["discards"]):
            discards_cnt[tid] += 1
        seen_tiles_cnt = np.add(seen_tiles_cnt, discards_cnt)
        discards_seq = np.zeros(33, dtype=np.int32)
        for j, tid in enumerate(state["players"][pid]["discards"]):
            discards_seq[j] = tid + 1
        if version == "0.1.1":
            encoded_state = np.concatenate([
                encoded_state, [pid], hand, exposed, discards_seq
            ]).astype(np.int32)
        elif version == "0.1.2":
            encoded_state = np.concatenate([
                encoded_state, [pid], hand, exposed, discards_cnt, [0]
            ]).astype(np.int32)
        elif version.startswith("0.2.0"):
            encoded_state = np.concatenate([
                encoded_state, [pid], hand, exposed, discards_cnt, discards_seq
            ]).astype(np.int32)
        else:
            raise NotImplementedError(f"Unsupported game encoding version: {version}")
    if version == "0.1.1":
        encoded_state = np.concatenate([encoded_state, [state["wall"]], state["option"]]).astype(np.int32)
    elif version == "0.1.2":
        encoded_state = np.concatenate([encoded_state, [state["wall"], 0], state["option"]]).astype(np.int32)
    elif version.startswith("0.2.0"):
        if analyzer is None:
            raise ValueError("Analyzer required for game state encoding")
        shanten, _, waits = analyzer(state["players"][player]["hand"])
        if version == "0.2.0a" and shanten > 1:
            waits = np.zeros(34, dtype=np.int32)
        encoded_state = np.concatenate([
            encoded_state, [state["wall"]], seen_tiles_cnt, state["option"], waits, [shanten]]
        ).astype(np.int32)

    if history:  # encode game history stat
        encoded_history = np.zeros((128, 78), dtype=np.int32)
        for i, (actor, action, tile, donor) in enumerate(state["actions"]):
            pid = (actor - player) % 4
            action_code = encode_action(action, tile, donor)
            encoded_history[i][0] = pid
            encoded_history[i][action_code + 1] = 1
        encoded_state = np.concatenate([encoded_state, encoded_history.flatten()])
    return encoded_state
