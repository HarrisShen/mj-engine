import random
import time
from mjengine.strategy import AnalyzerStrategy, tile_value
from mjengine.tiles import tiles_to_hand
from mjengine.utils import *


if __name__ == '__main__':
    n = 100_000
    samples = []
    wall = TID_LIST * 4
    tid_map = {old_tid: i for i, old_tid in enumerate(TID_LIST)}
    for _ in range(n):
        random.shuffle(wall)
        tiles = wall[:14]
        samples.append((tiles, tiles_to_hand([tid_map[tile] for tile in tiles])))

    results_old = [False for _ in range(n)]
    results_new = [False for _ in range(n)]
    print(f"Testing {n} samples")

    start = time.time()
    for i in range(n):
        tiles, hand = samples[i]
        results_old[i] = is_winning_old(tiles)
    finish = time.time()
    print(f"old 'is_winning': {finish - start:.3f}s")

    start = time.time()
    for i in range(n):
        tiles, hand = samples[i]
        results_new[i] = is_winning(hand)
    finish = time.time()
    print(f"new 'is_winning': {finish - start:.3f}s")

    print(all(results_old[i] == results_new[i] for i in range(n)))

    # Output
    # ============================================================================
    # Testing 100000 samples
    # old 'is_winning': 1.636s
    # new 'is_winning': 0.083s
    # True
