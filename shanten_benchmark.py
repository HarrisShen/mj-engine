import random
import time

from mjengine.shanten import Shanten
from mjengine.tiles import tiles_to_hand
from mjengine.utils import *


if __name__ == '__main__':
    n = 100
    samples = []
    tid_map = {old_tid: i for i, old_tid in enumerate(TID_LIST)}
    with open("samples/sampled_hands_13_100k.txt", "r") as f:
        for _ in range(n):
            tiles = [int(t) for t in f.readline().split()]
            samples.append((tiles, tiles_to_hand([tid_map[tile] for tile in tiles])))

    results_old = [0 for _ in range(n)]
    results_new = [0 for _ in range(n)]
    print(f"Testing {n} samples")

    start = time.time()
    for i in range(n):
        tiles, hand = samples[i]
        results_old[i] = distance_to_ready_old(tiles)
    finish = time.time()
    print(f"old method: {finish - start:.3f}s")

    sht = Shanten()
    sht.prepare("./")

    start = time.time()
    for i in range(n):
        tiles, hand = samples[i]
        results_new[i] = sht.calculate(hand, 4) - 1
    finish = time.time()
    print(f"new method: {finish - start:.3f}s")

    # for i in range(n):
    #     print(sorted([tid_map[tile] for tile in samples[i][0]]), results_old[i], results_new[i])

    n_correct = sum(results_old[i] == results_new[i] for i in range(n))
    accuracy = n_correct / n
    print(f"Correct results: {n_correct}")
    print(f"Accuracy of old method: {accuracy * 100:.2f}%")

    # Output
    # ============================================================================
    # Testing 1000 samples
    # old method: 43.717s
    # new method: 0.018s
    # Correct results: 388
    # Accuracy of old method: 38.80%
