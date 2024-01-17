import random
import string
from collections import Counter
from mjengine.constants import TID_LIST


def can_chow(hand: list[int], tile: int) -> list[bool]:
    result = [False for _ in range(4)]
    if tile > 27 or tile < 0:
        return result
    suit = tile // 9
    result[1] = (tile - 2) // 9 == suit and (tile - 1) // 9 == suit and hand[tile - 2] > 0 and hand[tile - 1] > 0
    result[2] = (tile - 1) // 9 == suit and (tile + 1) // 9 == suit and hand[tile - 1] > 0 and hand[tile + 1] > 0
    result[3] = (tile + 1) // 9 == suit and (tile + 2) // 9 == suit and hand[tile + 1] > 0 and hand[tile + 2] > 0
    result[0] = any(result[1:])
    return result


def can_pong(hand: list[int], tile: int) -> bool:
    return hand[tile] >= 2


def can_kong(hand: list[int], tile: int | None = None) -> bool:
    if tile is None:
        return max(hand) == 4
    if sum(hand) % 3 == 2:
        return hand[tile] == 4
    return hand[tile] == 3


def is_winning(hand: list[int], add_tile: int | None = None) -> bool:
    if add_tile is None:
        return _is_winning(hand)
    hand[add_tile] += 1
    result = _is_winning(hand)
    hand[add_tile] -= 1
    return result


def _is_winning(hand: list[int]) -> bool:
    head = -1
    for i in range(3):
        s = sum(hand[i * 9: (i + 1) * 9]) % 3
        if s == 1:
            return False
        if s == 2:
            if head == -1:
                head = i
            else:
                return False

    for i in range(27, 34):
        if hand[i] % 3 == 1:
            return False
        if hand[i] % 3 == 2:
            if head == -1:
                head = i
            else:
                return False

    for i in range(3):
        if i == head:
            if not is_winning_same_suit(hand[i * 9: (i + 1) * 9]):
                return False
        else:
            if not is_melds_same_suit(hand[i * 9: (i + 1) * 9]):
                return False

    return head != -1


def is_melds_same_suit(hand: list[int]) -> bool:
    """Return true if the hand is all melds (3N, sequences or triplets), False otherwise.
    The hand must only comprise one suit of tiles.
    Based on 'iswh0' from https://github.com/tomohxx/shanten-number/blob/master/judwin.cpp
    """
    if len(hand) != 9:
        raise ValueError("Hand of same suit only - expected length of hand is 9")
    a, b = hand[:2]
    for i in range(7):
        r = a % 3
        if b >= r and hand[i + 2] >= r:
            a, b = b - r, hand[i + 2] - r
        else:
            return False
    return a % 3 == 0 and b % 3 == 0


def is_winning_same_suit(hand: list[int]) -> bool:
    """Return true if the hand is winning (3N+2), False otherwise.
    The hand must only comprise one suit of tiles
    Based on 'iswh2' from https://github.com/tomohxx/shanten-number/blob/master/judwin.cpp
    """
    if len(hand) != 9:
        raise ValueError("Hand of same suit only - expected length of hand is 9")
    s = sum(i * v for i, v in enumerate(hand))
    for p in range(s * 2 % 3, 9, 3):
        if hand[p] >= 2:
            hand[p] -= 2
            if is_melds_same_suit(hand):
                hand[p] += 2
                return True
            else:
                hand[p] += 2
    return False


def is_winning_honor_only(hand: list[int]) -> bool:
    """Return true if the hand is winning (3N+2), False otherwise
    The hand must only have honor tiles
    Based on 'iswhs' from https://github.com/tomohxx/shanten-number/blob/master/judwin.cpp
    """
    head = -1
    for i in range(7):
        if hand[i] % 3 == 1:
            return False
        if hand[i] % 3 == 2:
            if head == -1:
                head = i
            else:
                return False
    return True


def is_winning_old(hand: list[int]) -> bool:
    """DEPRECATED
    Return True if the hand is a winning hand, False otherwise.
    A winning hand must have one pair of same tiles, with the rest of the tiles
    forming melds (triplets or sequences).
    """
    hand = [t for t in hand if t is not None]
    if len(hand) % 3 != 2:
        return False
    return _is_winning_old(Counter(hand))


def _is_winning_old(counter: Counter[int]) -> bool:
    """DEPRECATED
    Auxiliary function for is_winning.
    The counter is a Counter object that counts the number of each tile in the
    original hand.
    """
    for tid in counter:
        if counter[tid] > 4:
            return False
        if counter[tid] >= 2:
            new_counter = counter.copy()
            new_counter[tid] -= 2
            if _is_melds(new_counter):
                return True
    return False


def _is_melds(counter: Counter[int]) -> bool:
    """DEPRECATED
    Return True if the counter can form melds, False otherwise."""
    if not counter or counter.most_common(1)[0][1] == 0:
        return True
    for tid in counter:
        if counter[tid] == 0:
            continue
        if counter[tid] >= 3:
            new_counter = counter.copy()
            new_counter[tid] -= 3
            if _is_melds(new_counter):
                return True
        if counter[tid + 1] and counter[tid + 2]:
            new_counter = counter.copy()
            new_counter[tid] -= 1
            new_counter[tid + 1] -= 1
            new_counter[tid + 2] -= 1
            if _is_melds(new_counter):
                return True
    return False


def is_ready(hand: list[int]) -> bool:
    """Return True if the hand is ready, False otherwise.
    A ready hand is a hand that is exactly one tile away from winning.
    """
    for tile in range(34):
        hand[tile] += 1
        if is_winning(hand):
            hand[tile] -= 1
            return True
        hand[tile] -= 1
    return False


def screen_awaiting(hand: list[int]) -> list[int]:
    """Return a list of tiles that can make the hand winning.
    The list is empty if the hand is not ready.
    """
    return [tid for tid in TID_LIST if is_winning(hand + [tid])]


def distance_to_ready_old(hand: list[int]) -> int:
    """DEPRECATED
    Return the distance to win of the hand.
    The distance to win is the smallest number of changes needed to make
    so that the hand is ready
    """
    distance = 14
    counter = Counter(hand)
    memo = {}

    # 3 melds + 2 pairs
    pairs = [(tid, n) for tid, n in counter.most_common() if n >= 2]
    if len(pairs) >= 2:
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                new_counter = counter.copy()
                new_counter[pairs[i][0]] -= 2
                new_counter[pairs[j][0]] -= 2
                distance = min(distance, 1 + distance_to_melds(new_counter, memo))

    # 4 melds + 1 single
    for tid in counter:
        new_counter = counter.copy()
        new_counter[tid] -= 1
        distance = min(distance, 1 + distance_to_melds(new_counter, memo))
    return distance - 1


def distance_to_melds(hand: Counter, memo: dict | None = None) -> int:
    """Return the distance to melds of the hand.
    The distance to melds is the smallest number of changes to make the hand
    into all melds - without overlapping tiles between melds.
    """
    if memo is None:
        memo = {}

    def _get_distance(counter: Counter[int], n: int) -> int:
        if n == 0:
            return 0
        
        hand = tuple(counter_to_hand(counter, sort=True) + [n])
        if hand in memo:
            return memo[hand]
        
        if not counter or counter.most_common(1)[0][1] == 0:
            memo[hand] = 0
        else:
            distance = 14
            # triplets
            for tid in counter:
                if counter[tid] >= 3:
                    new_counter = counter.copy()
                    new_counter[tid] -= 3
                    distance = min(distance, _get_distance(new_counter, n - 1))
    
            # sequences
            for tid in TID_LIST:
                if tid > 40:
                    break
                # skip sequences without overlap with existing tiles
                if not (counter[tid] or counter[tid + 1] or counter[tid + 2]):
                    continue
                new_counter = counter.copy()
                seq_dist = 0
                for j in range(3):
                    if new_counter[tid + j] > 0:
                        new_counter[tid + j] -= 1
                    else:
                        seq_dist += 1
                if seq_dist >= distance:
                    continue
                distance = min(distance, seq_dist + _get_distance(new_counter, n - 1))
            memo[hand] = distance

        return memo[hand]

    # counter, n = Counter(hand), len(hand) // 3
    n = sum(hand.values()) // 3
    return _get_distance(hand, n)


def counter_to_hand(counter: Counter[int], sort=True, reverse=False) -> list[int]:
    """Return the hand represented by the counter.
    """
    hand = []
    for tid in counter:
        hand.extend([tid] * counter[tid])
    if sort:
        hand.sort(reverse=reverse)
    return hand
