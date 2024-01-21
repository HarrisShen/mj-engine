"""This file defines the tile representation for this module
The tid is an integer number from 0 to 33.
Number 0 is reserved for hidden tiles (backside-ups).
The naming (string representation) of tiles follows Japanese Mahjong tradition.
The first digit represents the suit, and the second digit represents the
rank (1-9).
The suits are:
0-8: Character/Manzu/M
9-17: Bamboo/Souzu/S
18-26: Dot/Pinzu/P
For Honors (Winds and Dragons), the numbers are 27-33 (1-7z).
The name is a string of the form "2m", "5s", "9p", "3z", in which the
first digit represents the rank (1-9), and the second digit represents
the suit (m/s/p/z).
"""


def tid_to_name(tid: int, suit_names: list[str] | None = None) -> str:
    """Return the name of the tile with the given tid.
    """
    if suit_names is None:
        suit_names = ["m", "s", "p", "z"]
    suit_id, rank_id = tid // 9, tid % 9 + 1
    return str(rank_id) + suit_names[suit_id]


def name_to_tid(name: str, suit_names: list[str] | None = None) -> int:
    """Return the tid of the tile with the given name.
    The name is a string of the form "2m", "5s", "9p", "3z", in which the
    first digit represents the rank (1-9), and the second digit represents
    the suit (m/s/p/z).
    """
    if suit_names is None:
        suit_names = ["m", "s", "p", "z"]
    rank_name, suit_name = name[0], name[1]
    if suit_name not in suit_names:
        raise ValueError(f"suit_name must be one of {suit_names}")
    suit_id = suit_names.index(suit_name)
    rank_id = int(rank_name)
    return suit_id * 9 + rank_id - 1


def tid_to_unicode(tid: int, flavor: str = "JP") -> str:
    """Return the unicode character of the tile with the given tid.
    For Dragons, the numbers are:
    32/5Z: White, 33/6Z: Green, 34/7Z: Red (JP - Japanese Mahjong)
    32/5Z: Red, 33/6Z: Green, 34/7Z: White (CN - Chinese Mahjong)
    The flavor is either "JP" or "CN". The default is "JP".
    The difference between the two flavors is the order of the Dragons, as shown above.
    """
    if flavor not in ["JP", "CN"]:
        raise ValueError("flavor must be either 'JP' or 'CN'")
    if tid < 27:
        return chr(0x1f007 + tid)
    if flavor == "JP":
        if tid == 31:
            return chr(0x1f006)
        if tid == 33:
            return chr(0x1f004)
    return chr(0x1f000 + tid - 27)


def hand_to_tiles(hand: list[int]) -> list[int]:
    tiles = []
    for i in range(len(hand)):
        tiles.extend([i for _ in range(hand[i])])
    return tiles


def tiles_to_hand(tiles: list[int]) -> list[int]:
    hand = [0 for _ in range(34)]
    for tile in tiles:
        hand[tile] += 1
    return hand


def tiles_left(game_dict: dict, mask: list[int | bool] | None = None):
    tiles = [4 for _ in range(34)]
    for i in range(4):
        player = game_dict["players"][i]
        hand = player["hand"]
        for j in range(34):
            tiles[j] -= hand[j]
        for tid in player["discards"]:
            tiles[tid] -= 1
        for meld in player["exposed"]:
            for tid in meld:
                tiles[tid] -= 1
    if mask is None:
        return tiles
    return [n if m else 0 for n, m in zip(tiles, mask)]
