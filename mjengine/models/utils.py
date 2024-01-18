from collections import deque
import random

import numpy as np

from mjengine.constants import PlayerAction


class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        return self.buffer[item]

    def __setitem__(self, key, value):
        self.buffer[key] = value

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done


def game_dict_to_numpy(state: dict, player: int | None = None) -> np.ndarray:
    # the state dict of game must be masked for opponents
    if player is None:
        for i in range(4):
            if sum(state["players"][i]["hand"]) > 0:
                player = i
                break
        else:
            raise ValueError("Game dict is not masked, please specify 'as_player' in 'Game.to_dict()'")
    encoded_state = np.array([])
    for i in range(4):
        pid = (player + i) % 4
        if i == 0:
            encoded_hand = np.array(state["players"][player]["hand"], dtype=np.int32)
        else:
            encoded_hand = np.array([])
        encoded_exposed = np.zeros(34, dtype=np.int32)
        for meld in state["players"][pid]["exposed"]:
            for tid in meld:
                encoded_exposed[tid] += 1
        encoded_discards = np.zeros(33, dtype=np.int32)
        for j, tid in enumerate(state["players"][pid]["discards"]):
            encoded_discards[j] = tid + 1
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
