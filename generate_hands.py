import argparse
import os.path
import random
from mjengine.strategy import ClosestReadyStrategy, tile_value
from mjengine.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--single", help="Generate a single hand with analysis", action="store_true")
    parser.add_argument("-n", "--size", help="Generate N hands", type=int)
    parser.add_argument("-N", "--handsize", help="Generate hand with size N", type=int, default=14)
    parser.add_argument("-r", "--seed", help="Random seed for generation", type=int)
    parser.add_argument("-e", "--encoding", help="The version of encoding, version 1 by default",
                        type=int, choices=[1, 2], default=1)
    parser.add_argument("-o", "--output", help="The path of the output file")
    args = parser.parse_args()

    r = random.Random()
    if args.seed is not None:
        print(f"random seed: {args.seed}")
        r = random.Random(int(args.seed))

    wall = TID_LIST * 4
    if args.encoding == 2:
        tid_map = {old_tid: i for i, old_tid in enumerate(TID_LIST)}
        wall = [tid_map[tile] for tile in wall]

    if args.single:
        r.shuffle(wall)
        hand = wall[:14]
        hand.sort()
        print(f"Hand: {hand}")
        dist_list = []
        for i in range(len(hand)):
            dist_list.append(distance_to_ready_old(hand[:i] + hand[i + 1:]))
        print("Distance to ready after discarding: ", dist_list)
        print("Tile value: ", [tile_value(hand, t) for t in hand])
        s = ClosestReadyStrategy("value")
        print(f"Discard: {hand[s(hand)[0]]}")

    if args.size:
        n = int(args.size)
        samples = []
        for _ in range(n):
            r.shuffle(wall)
            tiles = wall[:args.handsize]
            samples.append(tiles)
        with open(args.output, "w") as outf:
            for sample in samples:
                outf.write(" ".join(str(n) for n in sample) + "\n")
        print(f'Samples saved at "{os.path.abspath(args.output)}"')
