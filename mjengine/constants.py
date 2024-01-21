from enum import IntEnum


TID_LIST = [i * 10 + j for i in range(1, 4) for j in range(1, 10)] + \
    [40 + i for i in range(1, 8, 2)] + [50 + i for i in range(1, 6, 2)]
TID_SET = set(TID_LIST)


class PlayerAction(IntEnum):
    PASS = 0
    CHOW1 = 1  # chow tile is the largest tile in the sequence
    CHOW2 = 2  # chow tile is the middle tile in the sequence
    CHOW3 = 3  # chow tile is the smallest tile in the sequence
    PONG = 4
    KONG = 5
    WIN = 6


class GameStatus(IntEnum):
    START = 0
    DRAW = 1
    DISCARD = 2
    CHECK = 3
    END = 4
