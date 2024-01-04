from collections import Counter
from mjengine.constants import TID_LIST, TID_SET


def tid_to_name(tid: int) -> str:
    """Return the name of the tile with the given tid.
    The tid is a number between 11 and 55, inclusive.
    The first digit represents the suit, and the second digit represents the
    rank (1-9). The suits are:
    1: Bamboo
    2: Character
    3: Dot
    For Winds, the numbers are:
    41: East, 43: South, 45: West, 47: North
    For Dragons, the numbers are:
    51: Red, 53: Green, 55: White
    """
    suit_id, rank_id = tid // 10, tid % 10
    suit_name = {1: 'B', 2: 'C', 3: 'D', 4: 'W', 5: 'D'}
    if suit_id == 4:
        wind_name = {1: 'E', 3: 'S', 5: 'W', 7: 'N'}
        return suit_name[suit_id] + wind_name[rank_id]
    if suit_id == 5:
        dragon_name = {1: 'R', 3: 'G', 5: 'W'}
        return suit_name[suit_id] + dragon_name[rank_id]
    return suit_name[suit_id] + str(rank_id)


def name_to_tid(name: str) -> int:
    """Return the tid of the tile with the given name.
    The name is a string of the form "B1", "C5", "D9", "WE", "WS", "DR"
    """
    suit_name, rank_name = name[0], name[1:]
    suit_id = {'B': 1, 'C': 2, 'D': 3, 'W': 4, 'D': 5}[suit_name]
    if suit_id == 4:
        wind_id = {'E': 1, 'S': 3, 'W': 5, 'N': 7}[rank_name]
        return suit_id * 10 + wind_id
    if suit_id == 5:
        dragon_id = {'R': 1, 'G': 3, 'W': 5}[rank_name]
        return suit_id * 10 + dragon_id
    return suit_id * 10 + int(rank_name)


def tid_to_unicode(tid: int) -> str:
    """Return the unicode character of the tile with the given tid.
    """
    suit_id, rank_id = tid // 10, tid % 10
    if suit_id == 4:
        wind_unicode = {1: 0x1f000, 3: 0x1f001, 5: 0x1f002, 7: 0x1f003}
        return chr(wind_unicode[rank_id])
    if suit_id == 5:
        dragon_unicode = {1: 0x1f004, 3: 0x1f005, 5: 0x1f006}
        return chr(dragon_unicode[rank_id])
    return chr(0x1f007 + (suit_id - 1) * 9 + rank_id - 1)


def is_valid(tid: int) -> bool:
    """Return True if the tid is valid, False otherwise."""
    return tid in TID_SET


def is_valid_hand(hand: list[int]) -> bool:
    """Return True if the hand is valid, False otherwise.
    A valid hand must have 14 tiles, and each tile must be valid.
    """
    if hand % 3 == 0:
        return False
    return all(is_valid(tid) for tid in hand)


def is_winning(hand: list[int]) -> bool:
    """Return True if the hand is a winning hand, False otherwise.
    A winning hand must have one pair of same tiles, with the rest of the tiles
    forming melds (triplets or sequences).
    """
    if len(hand) % 3 != 2:
        return False
    return _is_winning(Counter(hand))


def can_chow(hand: list[int], tile: int) -> list[bool]:
    result = [False for _ in range(4)]
    tile_set = set(hand)
    if tile > 40 or tile not in tile_set:
        return result
    if tile - 2 in tile_set and tile - 1 in tile_set:
        result[0] = True
        result[1] = True
    if tile - 1 in tile_set and tile + 1 in tile_set:
        result[0] = True
        result[2] = True
    if tile + 1 in tile_set and tile + 2 in tile_set:
        result[0] = True
        result[3] = True
    return result


def can_pong(hand: list[int], tile: int) -> bool:
    return hand.count(tile) >= 2


def can_kong(hand: list[int], tile: int | None = None) -> bool:
    if tile is None:
        return Counter(hand).most_common(1)[0][1] == 4
    return hand.count(tile) >= 3


def _is_winning(counter: Counter[int]) -> bool:
    """Auxiliary function for is_winning.
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
    """Return True if the counter can form melds, False otherwise."""
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
    for tid in TID_LIST:
        if is_winning(hand + [tid]):
            return True
    return False


def screen_awaiting(hand: list[int]) -> list[int]:
    """Return a list of tiles that can make the hand winning.
    The list is empty if the hand is not ready.
    """
    return [tid for tid in TID_LIST if is_winning(hand + [tid])]


def distance_to_ready(hand: list[int]) -> int:
    """Return the distance to win of the hand.
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
                distance = min(distance, distance_to_melds(new_counter, memo))

    # 4 melds + 1 single
    for tid in counter:
        new_counter = counter.copy()
        new_counter[tid] -= 1
        distance = min(distance, 1 + distance_to_melds(new_counter, memo))
    return distance


def distance_to_melds(hand: list[int], memo: dict | None = None) -> int:
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

    counter, n = Counter(hand), len(hand) // 3
    return _get_distance(counter, n)


def counter_to_hand(counter: Counter[int], sort=True, reverse=False) -> list[int]:
    """Return the hand represented by the counter.
    """
    hand = []
    for tid in counter:
        hand.extend([tid] * counter[tid])
    if sort:
        hand.sort(reverse=reverse)
    return hand