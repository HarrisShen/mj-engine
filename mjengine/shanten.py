import os


class Shanten:
    def __init__(self):
        self.mp1 = []
        self.mp2 = []
        self.ret = []

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

    def _read_file(self, target: list[list[int]], file: str):
        with open(file, "r") as fin:
            for line in fin:
                target.append([int(n) for n in line.split()])

    def prepare(self, index_dir: str):
        self._read_file(self.mp1, os.path.join(index_dir, "index_s.txt"))
        self._read_file(self.mp2, os.path.join(index_dir, "index_h.txt"))

    def calculate(self, hand: list[int], m: int) -> int:
        """
        Calculate shanten number of the given hand
        """
        def acc_func(x, y):
            return x * 5 + y
        self.ret = self.mp1[accumulate(hand[1: 9], hand[0], acc_func)].copy()
        self._add1(self.mp1[accumulate(hand[10: 18], hand[9], acc_func)], m)
        self._add1(self.mp1[accumulate(hand[19: 27], hand[18], acc_func)], m)
        self._add2(self.mp2[accumulate(hand[28:], hand[27], acc_func)], m)
        return self.ret[m + 5]


def accumulate(iterable, init, func):
    result = init
    for n in iterable:
        result = func(result, n)
    return result
