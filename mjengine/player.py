from mjengine.constants import PlayerAction
from mjengine.option import Option
from mjengine.strategy import ClosestReadyStrategy, RandomStrategy, Strategy
from mjengine.utils import is_winning, tid_to_unicode


class Player:
    def __init__(self, strategy: Strategy = RandomStrategy()) -> None:
        self.game = None
        self.position = None
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
    
    def draw(self, tiles: list[int]) -> None:
        self.hand.extend(tiles)

    def decide(self, option: Option, last_discard: int | None = None) -> tuple[PlayerAction, int]:
        if last_discard is not None:
            action, _ = self.acquire(last_discard, option)
            return action, last_discard
        if option.discard:
            return None, self.select_discard()
        action, tile = self.examine(option)
        return action, tile
    
    def examine(self, option: Option | None = None) -> tuple[PlayerAction, int]:
        if len(self.hand) % 3 != 2:
            raise ValueError("Only hand after drawing can be examined")
        return self.strategy(self.hand, discard=False, option=option)
    
    def acquire(self, tile: int, option: Option) -> tuple[PlayerAction, int]:
        return self.strategy(self.hand, discard=False, tile=tile, option=option)
    
    def select_discard(self) -> int:
        _, tile = self.strategy(self.hand, discard=True)
        return tile
    
    def discard(self, tile: int) -> None:
        self.hand.remove(tile)
        self.discards.append(tile)

    def is_winning(self) -> bool:
        return is_winning(self.hand)
    
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

    def to_dict(self, hide_hand=False) -> dict:
        return {
            "hand": [0 for _ in range(len(self.hand))] if hide_hand else self.hand,
            "discards": self.discards,
            "exposed": self.exposed,
            "won": self.won,
            "score": self.score,
            "wins": self.wins,
            "self_wins": self.self_wins,
            "chucks": self.chucks,
        }


def make_player(strategy: str) -> Player:
    if strategy == "random":
        return Player(RandomStrategy())
    elif strategy == "closest":
        return Player(ClosestReadyStrategy())
    elif strategy == "closest_value":
        return Player(ClosestReadyStrategy("value"))
    