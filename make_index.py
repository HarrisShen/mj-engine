"""
This is the script for generating index for quick shanten number
(number of tiles to make hand ready) calculation.
Based on 'mkind1.cpp' from https://github.com/tomohxx/shanten-number/blob/master/mkind1.cpp
"""
from typing import Callable, TextIO, Iterator

from mjengine.utils import is_melds_same_suit, is_winning_same_suit, is_winning_honor_only


def dealtile(N: int) -> Callable:
    hd = [0 for _ in range(N)]

    def core(n: int, m: int, l: int, cnt: list[int], vec: list, func: Callable) -> None:
        if n >= N:
            if func(hd):
                cnt[0] += 1
                vec.append(hd.copy())
        else:
            lo, hi = max(0, m - l), min(4, m)
            for i in range(lo, hi + 1):
                hd[n] = i
                core(n + 1, m - i, l - 4, cnt, vec, func)

    return core


def deal(N: int) -> Callable:
    hd = [0 for _ in range(N)]
    sht = [0 for _ in range(10)]

    def core(n: int, kind: list, vec: list, fout: TextIO):
        if n >= N:
            calc(hd, sht, kind, iter(vec))
            fout.write(" ".join(str(int(n)) for n in sht) + "\n")
        else:
            for i in range(5):
                hd[n] = i
                core(n + 1, kind, vec, fout)
    return core


def calc(t: list[int], ret: list[int], kind: list, itr: Iterator[list[int]]):
    for i in range(10):
        sht = 100
        for j in range(kind[i][0]):
            tmp = int(inner_product(
                next(itr), t, 0,
                lambda x, y: x + y,
                lambda x, y: abs(x - y) + x - y) / 2)
            sht = min(tmp, sht)
        ret[i] = sht


def inner_product(l1: list, l2: list, init: int | float, op1: Callable, op2: Callable) -> float:
    assert len(l1) == len(l2)
    result = init
    for n1, n2 in zip(l1, l2):
        result = op1(result, op2(n1, n2))
    return result


def main():
    kind = [[0] for _ in range(10)]
    kind_h = [[0] for _ in range(10)]
    vec, vec7 = [], []
    for i in range(5):
        dealtile(9)(0, i * 3, 4 * 8, kind[i], vec, is_melds_same_suit)

    for i in range(5):
        dealtile(9)(0, i * 3 + 2, 4 * 8, kind[i + 5], vec, is_winning_same_suit)

    with open("index_s.txt", "w") as fout:
        deal(9)(0, kind, vec, fout)

    for i in range(5):
        dealtile(7)(0, i * 3, 4 * 6, kind_h[i], vec7, is_winning_honor_only)

    for i in range(5):
        dealtile(7)(0, i * 3 + 2, 4 * 6, kind_h[i + 5], vec7, is_winning_honor_only)

    with open("index_h.txt", "w") as fout2:
        deal(7)(0, kind_h, vec7, fout2)


if __name__ == "__main__":
    main()
