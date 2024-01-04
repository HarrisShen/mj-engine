import random
import time
from mjengine.strategy import ClosestReadyStrategy, tile_value
from mjengine.utils import *


if __name__ == '__main__':
    wall = TID_LIST * 4

    random.shuffle(wall)
    hand = wall[:14]
    # hand = [14, 18, 21, 24, 26, 33, 34, 39, 45, 55]
    hand.sort()
    print(f"Hand: {hand}")
    dist_list = []
    for i in range(len(hand)):
        dist_list.append(distance_to_ready(hand[:i] + hand[i + 1:]))
    print("Distance to ready after discarding: ", dist_list)
    print("Tile value: ", [tile_value(hand, t) for t in hand])
    s = ClosestReadyStrategy("value")
    print(f"Discard: {hand[s(hand)[0]]}")