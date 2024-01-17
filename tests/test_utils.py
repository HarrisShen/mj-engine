from mjengine.tiles import tiles_to_hand
from mjengine.utils import can_chow, is_winning_old, is_ready, can_pong, can_kong, is_melds_same_suit, \
    is_winning_same_suit, is_winning_honor_only, is_winning


def test_is_winning_old():
    assert is_winning_old([11, 11])
    assert is_winning_old([11, 11, 11, 12, 12])
    assert is_winning_old([11, 11, 11, 12, 13])
    assert is_winning_old([11, 11, 11, 12, 12, 12, 51, 51])
    assert is_winning_old([11, 11, 11, 11, 12, 13, 13, 13])

    # Not winning - wrong length
    assert not is_winning_old([11, 11, 11])
    assert not is_winning_old([11, 11, 12, 12])

    # Not winning - wrong tiles
    assert not is_winning_old([11, 11, 12, 13, 15])
    assert not is_winning_old([11, 12, 12, 13, 14])
    assert not is_winning_old([11, 11, 11, 12, 12, 12, 12])


def test_is_ready():
    assert is_ready([11])
    assert is_ready([11, 11, 11, 12, 12, 12, 12])
    assert is_ready([11, 12, 13, 14, 15, 16, 17])

    # Not ready
    assert not is_ready([11, 11])
    assert not is_ready([11, 11, 11, 22, 22, 22, 22])
    assert not is_ready([11, 11, 11, 11, 22, 22, 22])


def test_can_chow():
    tiles = [0, 1, 2, 3, 4, 5, 6]
    hand = tiles_to_hand(tiles)
    assert can_chow(hand, 3) == [True, True, True, True]
    assert can_chow(hand, 6) == [True, True, False, False]
    assert can_chow(hand, 0) == [True, False, False, True]
    assert can_chow(hand, 5) == [True, True, True, False]
    assert can_chow(hand, 1) == [True, False, True, True]
    assert can_chow(hand, 8) == [False, False, False, False]


def test_can_pong():
    hand = tiles_to_hand([0, 0, 1, 1, 1, 4, 5])
    assert can_pong(hand, 0)
    assert can_pong(hand, 1)
    assert not can_pong(hand, 5)


def test_can_kong():
    hand = tiles_to_hand([0, 0, 0, 0, 1, 1, 1])
    assert can_kong(hand)
    assert not can_kong(hand, 0)
    assert can_kong(hand, 1)
    assert not can_kong(hand, 3)
    hand = tiles_to_hand([0, 0, 0, 0, 1, 1, 1, 1])
    assert can_kong(hand)
    assert can_kong(hand, 0)
    assert can_kong(hand, 1)


def test_is_melds_same_suit():
    assert is_melds_same_suit(tiles_to_hand([])[:9])
    assert is_melds_same_suit(tiles_to_hand([0, 1, 2])[:9])
    assert not is_melds_same_suit(tiles_to_hand([3, 3, 5])[:9])
    assert is_melds_same_suit(tiles_to_hand([9, 11, 10])[9: 18])
    assert is_melds_same_suit(tiles_to_hand([0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4])[:9])
    assert not is_melds_same_suit(tiles_to_hand([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4])[:9])


def test_is_winning_same_suit():
    assert is_winning_same_suit(tiles_to_hand([0, 0])[:9])
    assert is_winning_same_suit(tiles_to_hand([1, 2, 3, 4, 4])[:9])
    assert not is_winning_same_suit(tiles_to_hand([1, 2, 3, 4, 5])[:9])
    assert is_winning_same_suit(tiles_to_hand([11, 11, 11, 12, 13])[9: 18])
    assert is_winning_same_suit(tiles_to_hand([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8])[:9])
    assert not is_winning_same_suit(tiles_to_hand([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 6, 7, 8, 8])[:9])


def test_is_winning_honor_only():
    assert is_winning_honor_only(tiles_to_hand([28, 28])[27:])
    assert is_winning_honor_only(tiles_to_hand([27, 27, 27, 28, 28])[27:])
    assert not is_winning_honor_only(tiles_to_hand([28, 29, 30, 31, 31])[27:])
    assert not is_winning_honor_only(tiles_to_hand([27, 27, 28, 28, 29, 29, 30, 30])[27:])


def test_is_winning():
    assert is_winning(tiles_to_hand([0, 0]))
    assert is_winning(tiles_to_hand([0, 0, 0, 1, 1]))
    assert is_winning(tiles_to_hand([0, 0, 0, 1, 2]))
    assert is_winning(tiles_to_hand([0, 0, 0, 1, 1, 1, 31, 31]))
    assert is_winning(tiles_to_hand([0, 0, 0, 0, 1, 2, 2, 2]))

    # Not winning - wrong length
    assert not is_winning(tiles_to_hand([0, 0, 0]))
    assert not is_winning(tiles_to_hand([0, 0, 1, 1]))

    # Not winning - wrong tiles
    assert not is_winning(tiles_to_hand([0, 0, 1, 2, 4]))
    assert not is_winning(tiles_to_hand([0, 1, 1, 2, 3]))
    assert not is_winning(tiles_to_hand([0, 0, 0, 0, 1, 1, 1, 1]))
