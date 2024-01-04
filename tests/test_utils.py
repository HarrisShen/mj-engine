from mjengine.utils import can_chow, is_winning, is_ready


def test_is_winning():
    assert is_winning([11, 11])
    assert is_winning([11, 11, 11, 12, 12])
    assert is_winning([11, 11, 11, 12, 13])
    assert is_winning([11, 11, 11, 12, 12, 12, 51, 51])
    assert is_winning([11, 11, 11, 11, 12, 13, 13, 13])

    # Not winning - wrong length
    assert not is_winning([11, 11, 11])
    assert not is_winning([11, 11, 12, 12])

    # Not winning - wrong tiles
    assert not is_winning([11, 11, 12, 13, 15])
    assert not is_winning([11, 12, 12, 13, 14])
    assert not is_winning([11, 11, 11, 12, 12, 12, 12])


def test_is_ready():
    assert is_ready([11])
    assert is_ready([11, 11, 11, 12, 12, 12, 12])
    assert is_ready([11, 12, 13, 14, 15, 16, 17])

    # Not ready
    assert not is_ready([11, 11])
    assert not is_ready([11, 11, 11, 22, 22, 22, 22])
    assert not is_ready([11, 11, 11, 11, 22, 22, 22])


def test_can_chow():
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 14) == [True, True, True, True]
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 17) == [True, True, False, False]
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 11) == [True, False, False, True]
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 16) == [True, True, True, False]
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 12) == [True, False, True, True]
    assert can_chow([11, 12, 13, 14, 15, 16, 17], 19) == [False, False, False, False]
