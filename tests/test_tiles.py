from pytest import fixture

from mjengine.tiles import tid_to_name, name_to_tid, tid_to_unicode, tiles_to_hand, hand_to_tiles, tiles_left


@fixture
def suits():
    return ["m", "s", "p", "z"]


@fixture
def names(suits):
    return tuple([str(i) + suit for suit in suits[:3] for i in range(1, 10)] +
                 [str(i) + suits[3] for i in range(1, 8)])


@fixture
def unicode_cn():
    return tuple([chr(0x1f007 + i) for i in range(27)] +
                 [chr(0x1f000 + i) for i in range(7)])


@fixture
def unicode_jp(unicode_cn):
    unicode_list = [chr(0x1f007 + i) for i in range(27)] + \
                   [chr(0x1f000 + i) for i in range(7)]
    unicode_list[-3], unicode_list[-1] = unicode_list[-1], unicode_list[-3]
    return tuple(unicode_list)


def test_tid_to_name(names):
    assert [tid_to_name(tid) for tid in range(34)] == list(names)


def test_name_to_tid(names):
    assert [name_to_tid(str(name)) for name in names] == list(range(34))


def test_tid_to_unicode(unicode_cn, unicode_jp):
    assert tuple(tid_to_unicode(tid, flavor="CN") for tid in range(34)) == unicode_cn
    assert tuple(tid_to_unicode(tid, flavor="JP") for tid in range(34)) == unicode_jp


def test_tiles_to_hand():
    tiles = list(range(34))
    assert tiles_to_hand(tiles) == [1 for _ in range(34)]


def test_hand_to_tiles():
    hand = [1 for _ in range(34)]
    tiles = hand_to_tiles(hand)
    for i in range(34):
        assert i in tiles


def test_tiles_left():
    tiles = tiles_left({
        "players": [{
            "hand": [0 for _ in range(34)],
            "discards": [],
            "exposed": []
        }] * 4
    })
    assert tiles == [4 for _ in range(34)]
    tiles = tiles_left({
        "players": [{
            "hand": [1] + [0 for _ in range(33)],
            "discards": [],
            "exposed": []
        }] * 4
    })
    assert tiles == [0] + [4 for _ in range(33)]
    tiles = tiles_left({
        "players": [{
            "hand": [0 for _ in range(34)],
            "discards": [],
            "exposed": []
        }] * 4
    }, [False] + [True for _ in range(33)])
    assert tiles == [0] + [4 for _ in range(33)]
