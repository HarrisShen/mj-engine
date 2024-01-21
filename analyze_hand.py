import argparse
import random

from mjengine.analyzer import Analyzer
from mjengine.tiles import tiles_to_hand, tid_to_unicode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tiles", help="Tiles to be analyzed", nargs="*")
    parser.add_argument("-r", "--random", help="Generate a random hand of given length", type=int)
    args = parser.parse_args()

    if args.tiles:
        tiles = [int(t) for t in args.tiles]
    elif args.random:
        wall = list(range(34)) * 4
        random.shuffle(wall)
        tiles = wall[:args.random]
    else:
        raise ValueError
    hand = tiles_to_hand(tiles)

    analyzer = Analyzer()
    analyzer.prepare("index")
    shanten, discard, wait = analyzer(hand)
    print(f"Hand: {' '.join(tid_to_unicode(t) for t in sorted(tiles))}")
    print(f"Shanten number: {shanten}")
    print(f"Tiles to discard: {' '.join(tid_to_unicode(t) for t, v in enumerate(discard) if v)}")
    print(f"Expecting tiles: {' '.join(tid_to_unicode(t) for t, v in enumerate(wait) if v)}")

    if sum(hand) % 3 != 2:
        exit(0)

    best_discards, max_n_exp = [], 0
    for t in range(34):
        if not discard[t]:
            continue
        print("\n" + "=" * 50 + "\n")
        print(f"Discard: {tid_to_unicode(t)}")
        hand[t] -= 1
        _, _, wait1 = analyzer(hand)
        n_exp = sum(wait1)
        print(f"Expecting {n_exp} tiles: {' '.join(tid_to_unicode(t1) for t1, v in enumerate(wait1) if v)}")
        if n_exp > max_n_exp:
            best_discards = [t]
            max_n_exp = n_exp
        elif n_exp == max_n_exp:
            best_discards.append(t)
        hand[t] += 1

    print("\n" + "=" * 50 + "\n")
    print(f"Recommendation: {' '.join(tid_to_unicode(t) for t in best_discards)}")
