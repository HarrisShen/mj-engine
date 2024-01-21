import os
from typing import Callable

from mjengine.tiles import tiles_left


class Singleton:
    def __new__(cls, *args, **kwargs):
        it = cls.__dict__.get("__it__")
        if it is not None:
            return it
        cls.__it__ = it = object.__new__(cls)
        it.init(*args, **kwargs)
        return it

    def init(self, *args, **kwargs):
        pass


def _read_file(target: list[list[int]], file: str, proc_func: Callable = None):
    with open(file, "r") as fin:
        for line in fin:
            parsed_line = [int(n) for n in line.split()]
            if proc_func is not None:
                parsed_line = proc_func(parsed_line)
            target.append(parsed_line)


def accumulate(iterable, init, func):
    result = init
    for n in iterable:
        result = func(result, n)
    return result


class Shanten:
    def __init__(self):
        self.mp1 = []
        self.mp2 = []
        self.ret = []

    def prepare(self, index_dir: str = "index"):
        _read_file(self.mp1, os.path.join(index_dir, "index_s.txt"))
        _read_file(self.mp2, os.path.join(index_dir, "index_h.txt"))

    def _add1(self, rhs, m: int) -> None:
        for j in range(m + 5, 4, -1):
            sht = min(self.ret[j] + rhs[0], self.ret[0] + rhs[j])
            for k in range(5, j):
                sht = min(sht, self.ret[k] + rhs[j - k], self.ret[j - k] + rhs[k])
            self.ret[j] = sht
        for j in range(m, -1, -1):
            sht = self.ret[j] + rhs[0]
            for k in range(j):
                sht = min(sht, self.ret[k] + rhs[j - k])
            self.ret[j] = sht

    def _add2(self, rhs, m: int) -> None:
        j = m + 5
        sht = min(self.ret[j] + rhs[0], self.ret[0] + rhs[j])
        for k in range(5, j):
            sht = min(sht, self.ret[k] + rhs[j - k], self.ret[j - k] + rhs[k])
        self.ret[j] = sht

    def calculate(self, hand: list[int], m: int) -> int:
        """
        Calculate substitute number of the given hand
        Note that the raw values in 'ret' list are substitute number (Shanten number + 1)
        """
        def acc_func(x, y):
            return x * 5 + y
        self.ret = self.mp1[accumulate(hand[1: 9], hand[0], acc_func)].copy()
        self._add1(self.mp1[accumulate(hand[10: 18], hand[9], acc_func)], m)
        self._add1(self.mp1[accumulate(hand[19: 27], hand[18], acc_func)], m)
        self._add2(self.mp2[accumulate(hand[28:], hand[27], acc_func)], m)
        return self.ret[m + 5]

    def __call__(self, hand):
        """
        Calculate the Shanten number
        """
        m = sum(hand) // 3
        return self.calculate(hand, m) - 1


def _shift(
        l: tuple[int, int, int],
        r: tuple[int, int, int]) -> tuple[int, int, int]:
    lv, lx, ly = l
    rv, rx, ry = r
    if lv == rv:
        lx |= rx
        ly |= ry
    elif lv > rv:
        lv = rv
        lx = rx
        ly = ry
    return lv, lx, ly


class Analyzer(Singleton):

    def init(self):
        self.mp1 = []
        self.mp2 = []
        self.ret = []

    def __call__(self, hand: list[int]) -> tuple[int, list[bool], list[bool]]:
        m = sum(hand) // 3
        sht, disc, wait = self.calculate(hand, m)
        disc_bits, wait_bits = [False for _ in range(34)], [False for _ in range(34)]
        for i in range(34):
            disc_bits[i] = bool(disc & 1)
            disc >>= 1
            wait_bits[i] = bool(wait & 1)
            wait >>= 1
        return sht, disc_bits, wait_bits

    def prepare(self, index_dir: str = "index") -> None:
        if self.mp1:
            return

        def expand_index(nums: list[int]) -> list[int]:
            assert len(nums) == 10
            nums += [0 for _ in range(20)]
            for i in range(10):
                n = nums[i]
                nums[i] = n & ((1 << 4) - 1)
                nums[i + 10] = (n >> 4) & ((1 << 9) - 1)
                nums[i + 20] = (n >> 13) & ((1 << 9) - 1)
            return nums

        _read_file(self.mp1, os.path.join(index_dir, "index_dw_s.txt"), expand_index)
        _read_file(self.mp2, os.path.join(index_dir, "index_dw_h.txt"), expand_index)

    def _add1(self, rhs: list[int], m: int):
        for j in range(m + 5, 4, -1):
            sht = self.ret[j] + rhs[0]
            disc = (self.ret[j + 10] << 9) | rhs[10]
            wait = (self.ret[j + 20] << 9) | rhs[20]

            sht, disc, wait = _shift(
                (sht, disc, wait),
                (
                    self.ret[0] + rhs[j],
                    (self.ret[10] << 9) | rhs[j + 10],
                    (self.ret[20] << 9) | rhs[j + 20]
                )
            )

            for k in range(5, j):
                sht, disc, wait = _shift(
                    (sht, disc, wait),
                    (
                        self.ret[k] + rhs[j - k],
                        (self.ret[k + 10] << 9) | rhs[j - k + 10],
                        (self.ret[k + 20] << 9) | rhs[j - k + 20]
                    )
                )
                sht, disc, wait = _shift(
                    (sht, disc, wait),
                    (
                        self.ret[j - k] + rhs[k],
                        (self.ret[j - k + 10] << 9) | rhs[k + 10],
                        (self.ret[j - k + 20] << 9) | rhs[k + 20]
                    )
                )

            self.ret[j] = sht
            self.ret[j + 10] = disc
            self.ret[j + 20] = wait

        for j in range(m, -1, -1):
            sht = self.ret[j] + rhs[0]
            disc = (self.ret[j + 10] << 9) | rhs[10]
            wait = (self.ret[j + 20] << 9) | rhs[20]

            for k in range(j):
                sht, disc, wait = _shift(
                    (sht, disc, wait),
                    (
                        self.ret[k] + rhs[j - k],
                        (self.ret[k + 10] << 9) | rhs[j - k + 10],
                        (self.ret[k + 20] << 9) | rhs[j - k + 20]
                    )
                )

            self.ret[j] = sht
            self.ret[j + 10] = disc
            self.ret[j + 20] = wait

    def _add2(self, rhs: list[int], m: int):
        j = m + 5
        sht = self.ret[j] + rhs[0]

        disc = (self.ret[j + 10] << 9) | rhs[10]
        wait = (self.ret[j + 20] << 9) | rhs[20]
        sht, disc, wait = _shift(
            (sht, disc, wait),
            (
                self.ret[0] + rhs[j],
                (self.ret[10] << 9) | rhs[j + 10],
                (self.ret[20] << 9) | rhs[j + 20]
            )
        )

        for k in range(5, j):
            sht, disc, wait = _shift(
                (sht, disc, wait),
                (
                    self.ret[k] + rhs[j - k],
                    (self.ret[k + 10] << 9) | rhs[j - k + 10],
                    (self.ret[k + 20] << 9) | rhs[j - k + 20]
                )
            )
            sht, disc, wait = _shift(
                (sht, disc, wait),
                (
                    self.ret[j - k] + rhs[k],
                    (self.ret[j - k + 10] << 9) | rhs[k + 10],
                    (self.ret[j - k + 20] << 9) | rhs[k + 20]
                )
            )

        self.ret[j] = sht
        self.ret[j + 10] = disc
        self.ret[j + 20] = wait

    def calculate(self, hand: list[int], m: int):
        def acc_func(x, y):
            return x * 5 + y
        self.ret = self.mp2[accumulate(hand[28:], hand[27], acc_func)].copy()

        self._add1(self.mp1[accumulate(hand[19: 27], hand[18], acc_func)], m)
        self._add1(self.mp1[accumulate(hand[10: 18], hand[9], acc_func)], m)

        self._add2(self.mp1[accumulate(hand[1: 9], hand[0], acc_func)], m)

        return self.ret[m + 5], self.ret[m + 15], self.ret[m + 25]

    def best_discard(self, hand: list[int], game_state: dict | None = None):
        _, discard, _ = self(hand)
        best_discards, max_n_exp = [], 0
        for t in range(len(hand)):
            if not discard[t]:
                continue
            hand[t] -= 1
            _, _, wait1 = self(hand)
            n_exp = sum(wait1) if game_state is None else sum(tiles_left(game_state, mask=wait1))
            if n_exp > max_n_exp:
                best_discards = [t]
                max_n_exp = n_exp
            elif n_exp == max_n_exp:
                best_discards.append(t)
            hand[t] += 1
        return best_discards, max_n_exp
