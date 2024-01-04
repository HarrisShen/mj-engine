import random
import time
from mjengine.utils import *


if __name__ == '__main__':
    wall = TID_LIST * 4
    start = time.time()
    n = 2000
    for i in range(n):
        random.shuffle(wall)
        hand = wall[:13]
        hand.sort()

        print(f"({i + 1}) Hand: {hand}")
        print(f"Distance to ready: {distance_to_ready(hand)}")
    
    end = time.time()
    print(f"Total time: {end - start:.3f}s")
    print(f"Average time: {1000 * (end - start) / n:.1f}ms")
