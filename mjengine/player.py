from mjengine.constants import PlayerAction
from mjengine.strategy import ClosestReadyStrategy, RandomStrategy, Strategy
from mjengine.utils import is_winning, tid_to_unicode


class Player:
    def __init__(self, strategy: Strategy = RandomStrategy()) -> None:
        self.hand = []
        self.discards = []
        self.exposed = []
        self.won = False
        self.strategy = strategy

        # player stats
        self.score = 0
        self.wins = 0
        self.self_wins = 0
        self.chucks = 0

    def reset(self) -> None:
        self.hand = []
        self.discards = []
        self.exposed = []
        self.won = False

    def hand_to_str(self, sort=True) -> str:
        if sort:
            return ' '.join([tid_to_unicode(tid) for tid in sorted(self.hand)])
        return ' '.join([tid_to_unicode(tid) for tid in self.hand])
    
    def draw(self, wall: list[int], n: int=1) -> list[int]:
        self.hand += wall[:n]
        del wall[:n]
        return self.hand[-n:]
    
    def examine(self) -> PlayerAction:
        if len(self.hand) % 3 != 2:
            raise ValueError("Only hand after drawing can be examined")
        return self.strategy(self.hand)

    def discard(self) -> int:
        index, tile = self.strategy(self.hand, discard=True)
        self.hand = self.hand[:index] + self.hand[index + 1 :]
        self.discards.append(tile)
        return tile

    def is_winning(self) -> bool:
        return is_winning(self.hand)
    
    def acquire(self, tile: int, options: list[bool]) -> PlayerAction:
        return self.strategy(self.hand, tile=tile, options=options)[0]
    
    def chow(self, tile: int, mode: int) -> None:
        if mode == PlayerAction.CHOW1:
            self.hand.remove(tile - 2)
            self.hand.remove(tile - 1)
            self.exposed.append([tile - 2, tile - 1, tile])
        elif mode == PlayerAction.CHOW2:
            self.hand.remove(tile - 1)
            self.hand.remove(tile + 1)
            self.exposed.append([tile - 1, tile, tile + 1])
        elif mode == PlayerAction.CHOW3:
            self.hand.remove(tile + 1)
            self.hand.remove(tile + 2)
            self.exposed.append([tile, tile + 1, tile + 2])
    
    def pong(self, tile: int) -> None:
        self.hand.remove(tile)
        self.hand.remove(tile)
        self.exposed.append([tile, tile, tile])
    
    def kong(self, tile: int) -> None:
        self.hand = [tid for tid in self.hand if tid != tile]
        self.exposed.append([tile, tile, tile, tile])


def make_player(strategy: str) -> Player:
    if strategy == "random":
        return Player(RandomStrategy())
    elif strategy == "closest":
        return Player(ClosestReadyStrategy())
    elif strategy == "closest_value":
        return Player(ClosestReadyStrategy("value"))
    