import os

from mjengine.constants import PlayerAction
from mjengine.option import Option
from mjengine.strategy import AnalyzerStrategy, RandomStrategy, Strategy, RLAgentStrategy
from mjengine.tiles import tid_to_unicode, hand_to_tiles
from mjengine.utils import is_winning


class Player:
    def __init__(self, strategy: Strategy) -> None:
        self.game = None
        self.position = None
        self.hand = [0 for _ in range(34)]
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
        self.hand = [0 for _ in range(34)]
        self.discards = []
        self.exposed = []
        self.won = False

    def hand_to_str(self) -> str:
        return ' '.join([tid_to_unicode(tid) for tid in hand_to_tiles(self.hand)])
    
    def draw(self, tiles: list[int]) -> None:
        for tile in tiles:
            self.hand[tile] += 1

    def decide(
            self,
            option: Option,
            last_discard: int | None,
            game_info: dict) -> tuple[PlayerAction | None, int]:
        if last_discard is not None:
            action, _ = self.acquire(last_discard, option, game_info)
            return action, last_discard
        if option.discard:
            return None, self.select_discard(game_info)
        action, tile = self.examine(option, game_info)
        return action, tile
    
    def examine(self, option: Option, info: dict) -> tuple[PlayerAction, int]:
        if sum(self.hand) % 3 != 2:
            raise ValueError("Only hand after drawing can be examined")
        return self.strategy(self.hand, info, discard=False, option=option)
    
    def acquire(self, tile: int, option: Option, info: dict) -> tuple[PlayerAction, int]:
        return self.strategy(self.hand, info, discard=False, tile=tile, option=option)
    
    def select_discard(self, info: dict) -> int:
        _, tile = self.strategy(self.hand, info, discard=True)
        return tile
    
    def discard(self, tile: int) -> None:
        self.hand[tile] -= 1
        self.discards.append(tile)

    def is_winning(self) -> bool:
        return is_winning(self.hand)
    
    def chow(self, tile: int, mode: int) -> None:
        if mode == PlayerAction.CHOW1:
            self.hand[tile - 1] -= 1
            self.hand[tile - 2] -= 1
            self.exposed.append([tile - 2, tile - 1])
        elif mode == PlayerAction.CHOW2:
            self.hand[tile - 1] -= 1
            self.hand[tile + 1] -= 1
            self.exposed.append([tile - 1, tile + 1])
        elif mode == PlayerAction.CHOW3:
            self.hand[tile + 1] -= 1
            self.hand[tile + 2] -= 1
            self.exposed.append([tile + 1, tile + 2])
    
    def pong(self, tile: int) -> None:
        self.hand[tile] -= 2
        self.exposed.append([tile, tile])
    
    def kong(self, tile: int) -> None:
        self.exposed.append([tile for _ in range(self.hand[tile])])
        self.hand[tile] = 0

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

    def summary(self, games: int | None = None) -> dict:
        player_summary = {
            "score": self.score,
            "wins": self.wins,
            "self_wins": self.wins,
            "chucks": self.chucks
        }
        if games is None:
            return player_summary
        self_win_rate = 0
        if self.wins:
            self_win_rate = self.self_wins / self.wins
        player_summary["avg_score"] = self.score / games
        player_summary["win_rate"] = self.wins / games
        player_summary["self_win_rate"] = self_win_rate
        player_summary["chuck_rate"] = self.chucks / games
        return player_summary


def make_player(strategy: str) -> Player:
    if strategy in ["random", "random0", "r", "r0"]:
        return Player(RandomStrategy(0))
    elif strategy in ["random1", "r1"]:
        return Player(RandomStrategy(1))
    elif strategy == "analyzer" or strategy == "a":
        return Player(AnalyzerStrategy())
    elif strategy == "analyzer_value" or strategy == "value" or strategy == "v":
        return Player(AnalyzerStrategy("value"))
    elif strategy == "analyzer_exp0" or strategy == "exp0" or strategy == "e0":
        return Player(AnalyzerStrategy("exp0"))
    elif strategy in ["analyzer_exp1", "exp1", "e1", "e"]:
        return Player(AnalyzerStrategy("exp1"))
    elif os.path.exists(strategy):
        import torch

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        return Player(RLAgentStrategy.load(model_path=strategy, device=device))

    raise ValueError("Invalid strategy")


def make_players(strats: list[str]) -> list[Player]:
    return [make_player(strategy) for strategy in strats]
