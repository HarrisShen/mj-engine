import random
import time
from mjengine.strategy import ClosestReadyStrategy, tile_value
from mjengine.tiles import tiles_to_hand
from mjengine.utils import *


if __name__ == '__main__':
    wall = TID_LIST * 4
    tid_map = {old_tid: i for i, old_tid in enumerate(TID_LIST)}
    random.shuffle(wall)
    hand = wall[:14]

    hand.sort()
    print(f"Hand: {hand}")
    dist_list = []
    for i in range(len(hand)):
        dist_list.append(distance_to_ready(hand[:i] + hand[i + 1:]))
    print("Distance to ready after discarding: ", dist_list)
    print("Tile value: ", [tile_value(hand, t) for t in hand])
    s = ClosestReadyStrategy("value")
    print(f"Discard: {hand[s(hand)[0]]}")
