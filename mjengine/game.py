import logging
import random
from collections import deque

import numpy as np

from mjengine.constants import GameStatus, PlayerAction
from mjengine.models.utils import game_dict_to_numpy
from mjengine.option import Option
from mjengine.player import Player
from mjengine.tiles import tid_to_unicode
from mjengine.utils import can_chow, can_kong, can_pong, is_winning


class Game:
    def __init__(
            self, 
            players: list[Player] | None = None, 
            round_limit: int | None = None,
            game_limit: int | None = None,
            seed: int | float | None = None,
            verbose: int = 0) -> None:
        self.round_limit = round_limit
        self.game_limit = game_limit
        if self.round_limit is not None and self.game_limit is not None:
            raise ValueError("Cannot set both round limit and game limit")
        if self.game_limit is not None and self.game_limit < 1:
            raise ValueError("Invalid game limit")
        if self.round_limit is None and self.game_limit is None:
            self.round_limit = 1

        self.wall = []
        self.dealer = 0
        self.round = 0
        self.games = 0
        self.status = GameStatus.START
        if players is None:
            players = [Player() for _ in range(4)]
        self.players = players
        self.register_players()
        self.current_player = 0
        self.acting_player = -1
        self.option = None
        self.acting_queue = None
        self.waiting = []

        if seed is not None:
            random.seed(seed)

        if verbose == 2:
            logging.basicConfig(level=logging.DEBUG)
        elif verbose == 1:
            logging.basicConfig(level=logging.INFO)
        elif verbose == 0:
            logging.basicConfig(level=logging.WARNING)
        else:
            raise ValueError("verbose level can only be 0, 1 or 2")

    def register_players(self) -> None:
        for i in range(len(self.players)):
            self.players[i].position = i

    def reset(self) -> None:
        self.wall = []
        self.status = GameStatus.START
        for i in range(4):
            self.players[i].reset()

    def start_game(self) -> None:
        if self.status != GameStatus.START:
            raise ValueError("Game is not ready to deal")
        
        self.wall = list(range(34)) * 4
        random.shuffle(self.wall)

        # mimic real procedure - deal 4 tiles to each player, then 1 tile to each player
        for _ in range(3):
            for i in range(4):
                self.deal((i + self.dealer) % 4, 4)
        for i in range(4):
            self.deal((i + self.dealer) % 4, 1)
            logging.debug(f"Player {(i + self.dealer) % 4}: {self.players[(i + self.dealer) % 4].hand_to_str()}")

        self.current_player = self.dealer
        self.status = GameStatus.DRAW

    def deal(self, player: int, n: int = 1) -> list[int]:
        if n > len(self.wall):
            raise ValueError("Insufficient tiles in the wall")
        tiles = self.wall[-n:]
        self.wall = self.wall[:-n]
        self.players[player].draw(tiles)
        return tiles

    """
    Game class as a host of the game, which is responsible for the game flow.
    1. Deal tiles to players
    2. Check whether any player wins
    3. Check whether any player can chow, pong, kong, chuck
    4. Ask players to make decisions
    5. Apply decisions
    6. Repeat 2-5 until someone wins or wall is empty
    7. Settle scores
    Separate the logic of players decision making from the game flow.
    The goal is the Game gives the player a list of options, and the player makes a decision.
    When the player makes a decision, the game decides whether the decision is valid,
    and whether the action can be applied to the game, since other players may have made
    decisions that have the same or higher priority - e.g. multiple players can win by chuck.
    """
    def play(self) -> None:
        while not self.is_finished():
            self.reset()
            logging.info(f"Round {self.round} - Dealer is Player {self.dealer} now.")
            self.start_game()
            while self.status < GameStatus.END:
                self.get_option()
                last_discard = None
                if self.status == GameStatus.CHECK:
                    last_discard = self.players[self.current_player].discards[-1]
                action, tile = self.players[self.acting_player].decide(
                    self.option,
                    last_discard,
                    self.to_dict(self.acting_player))
                self.apply_action(action, tile)
            self.settle_score()
            if not self.players[self.dealer].is_winning():
                self.dealer = (self.dealer + 1) % 4
                # end of a round
                if self.dealer == 0:
                    self.round += 1
            self.games += 1
            logging.info(self.score_summary(detail=0))

    def is_finished(self) -> bool:
        if self.round_limit is not None and self.round >= self.round_limit:
            return True
        if self.game_limit is not None and self.games >= self.game_limit:
            return True
        return False

    def get_option(self) -> None:
        """
        Return:
            player: the player who should make decision
            option: the option for the player to make decision
            tile: the tile to be acquired, or None if the player is playing their own turn
        """
        if self.status == GameStatus.START:
            raise ValueError("Game is not ready to play, please deal first")
        
        # draw a tile and check winning
        if self.status == GameStatus.DRAW:
            last_draw = self.deal(self.current_player, 1)[0]
            logging.debug(f"Player {self.current_player} draws {tid_to_unicode(last_draw)} " 
                          f"(Tiles in wall - {len(self.wall)})")
            logging.debug(f"Player {self.current_player}: {self.players[self.current_player].hand_to_str()}")
            hand = self.players[self.current_player].hand
            kong_tiles = [tid for tid in range(len(hand)) if hand[tid] == 4]
            if not self.wall:  # unable to kong if no tile to draw
                kong_tiles = []
            self_win = is_winning(hand)
            if self_win or kong_tiles:
                option = Option(concealed_kong=kong_tiles, win_from_self=self_win)
                self.acting_player = self.current_player
                self.option = option
                return
            self.status = GameStatus.DISCARD

        # discard a tile
        if self.status == GameStatus.DISCARD:
            option = Option(discard=True, hand=self.players[self.current_player].hand)
            self.acting_player = self.current_player
            self.option = option
            return

        # check chow, pong, kong and chuck (winning from others' discards)
        last_discard = self.players[self.current_player].discards[-1]
        if self.acting_queue is None:
            self.acting_queue = deque()
            options = []
            for i in range(1, 4):
                pid = (self.current_player + i) % 4
                hand = self.players[pid].hand
                option = Option(
                    chow=can_chow(hand, last_discard) if i == 1 else [False, False, False, False],
                    pong=can_pong(hand, last_discard),
                    exposed_kong=can_kong(hand, last_discard) and self.wall,
                    win_from_chuck=is_winning(hand, last_discard)
                )
                if option.tier() > (0, 0):
                    options.append((pid, option))
            options.sort(key=lambda x: x[1].tier(), reverse=True)
            self.acting_queue.extend(options)
        # no one can chow, pong, kong or chuck, go back to draw
        if not self.acting_queue:
            self.acting_queue = None
            self.current_player = (self.current_player + 1) % 4
            self.status = GameStatus.DRAW
            self.get_option()
            return
        self.acting_player, self.option = self.acting_queue.popleft()

    def apply_action(self, action: PlayerAction, tile: int | None) -> None:
        player = self.acting_player
        if action is None:
            if self.status != GameStatus.DISCARD:
                raise ValueError("Invalid action")
            if self.players[player].hand[tile] < 1:
                raise ValueError(f"Invalid tile (tid {tile}) to discard")
            self.players[player].discard(tile)
            logging.debug(f"Player {player} discards {tid_to_unicode(tile)}")
            logging.debug(f"Player {player}: {self.players[player].hand_to_str()}")
            if not self.wall:
                logging.info("Wall is empty!")
                self.status = GameStatus.END
                return
            self.status = GameStatus.CHECK
            return
        
        hand = self.players[player].hand
        if action == PlayerAction.WIN:
            if not is_winning(hand) and not is_winning(hand, tile):
                raise ValueError("Invalid action")
        elif action == PlayerAction.KONG:
            if tile is None or not can_kong(self.players[player].hand, tile):
                raise ValueError("Invalid action")
        elif action == PlayerAction.PONG:
            if tile is None or not can_pong(self.players[player].hand, tile):
                raise ValueError("Invalid action")
        elif action > PlayerAction.PASS:  # chow
            if tile is None or not can_chow(self.players[player].hand, tile)[0]:
                raise ValueError("Invalid action")
        self.waiting.append((player, action))

        # check whether the action can be taken right away
        if self.acting_queue and action <= self.acting_queue[0][1].highest_action():
            # need to wait
            return
        # apply the action
        self.acting_queue = None  # reset acting queue
        self.waiting.sort(key=lambda x: x[1], reverse=True)
        action = self.waiting[0][1]
        acting_players = [i for i, a in self.waiting if a == action]
        self.waiting = []

        if action == PlayerAction.WIN:
            for i in acting_players:
                self.players[i].won = True
                if tile is not None:
                    self.players[i].hand[tile] += 1
                logging.info(f"Player {i} wins!")
                logging.info(f"Player {i}: {self.players[i].hand_to_str()}")
            self.status = GameStatus.END
            return
        
        if action == PlayerAction.KONG:
            self.players[acting_players[0]].kong(tile)
            logging.debug(f"Player {acting_players[0]} kongs!")
            self.current_player = acting_players[0]
            self.status = GameStatus.DRAW
            return
        
        if action == PlayerAction.PONG:
            self.players[acting_players[0]].pong(tile)
            logging.debug(f"Player {acting_players[0]} pongs!")
            self.current_player = acting_players[0]
            self.status = GameStatus.DISCARD
            return
        
        if action > PlayerAction.PASS:  # chow
            self.players[acting_players[0]].chow(tile, action)
            logging.debug(f"Player {acting_players[0]} chows!")
            self.current_player = acting_players[0]
            self.status = GameStatus.DISCARD
            return
        
        if action == PlayerAction.PASS:
            # all pass after check
            if self.status == GameStatus.CHECK:
                self.current_player = (self.current_player + 1) % 4
                self.status = GameStatus.DRAW
                return
            # passed after draw
            if self.status == GameStatus.DRAW:
                self.status = GameStatus.DISCARD
                return

        raise ValueError("Invalid action")

    def settle_score(self) -> None:
        # no one wins
        if all([not self.players[i].won for i in range(4)]):
            return
        
        # win by self-draw
        if self.players[self.current_player].won:
            self.players[self.current_player].wins += 1
            self.players[self.current_player].self_wins += 1
            for i in range(4):
                if i == self.current_player:
                    continue
                self.players[i].score -= 1
            self.players[self.current_player].score += 3
            return
        
        # win by chuck
        self.players[self.current_player].chucks += 1
        for i in range(4):
            if self.players[i].won:
                self.players[i].wins += 1
                self.players[i].score += 1
                self.players[self.current_player].score -= 1

    def score_summary(self, detail=0) -> str:
        """Produce the score summary of the game in a table format.
        """
        if detail == 0:
            report = f"\nP\tS\tW\tSW\tC\n"
            for i in range(4):
                report += f"{i}\t{self.players[i].score}\t{self.players[i].wins}\t"\
                          f"{self.players[i].self_wins}\t{self.players[i].chucks}\n"
            return report
        if detail == 1:
            report = f"\nPos\tScore\tWin\tSelf W\tChuck\tAvg S\tWR\tSWR\tCR\n"
            for i in range(4):
                self_win_rate = 0
                if self.players[i].wins:
                    self_win_rate = self.players[i].self_wins / self.players[i].wins
                report += f"{i}\t{self.players[i].score}\t{self.players[i].wins}\t"\
                          f"{self.players[i].self_wins}\t{self.players[i].chucks}\t"\
                          f"{self.players[i].score / self.games:.4f}\t"\
                          f"{self.players[i].wins / self.games * 100:.3f}\t" \
                          f"{self_win_rate * 100:.3f}\t" \
                          f"{self.players[i].chucks / self.games * 100:.3f}\n"
            return report

    def tiles_left(self, as_player: int, mask: list[bool] | None = None) -> list[int]:
        tiles = [4 for _ in range(34)]
        for i in range(4):
            if i == as_player:
                for tid, cnt in enumerate(self.players[i].hand):
                    tiles[tid] -= cnt
            for tid in self.players[i].discards:
                tiles[tid] -= 1
            for meld in self.players[i].exposed:
                for tid in meld:
                    tiles[tid] -= 1
        if mask is None:
            return tiles
        return [cnt if bv else 0 for cnt, bv in zip(tiles, mask)]
            
    def to_dict(self, as_player: int | str | None = None) -> dict:
        """Return a dict representation of the game.
        """
        if isinstance(as_player, str):
            if as_player == "acting":
                as_player = self.acting_player
            elif as_player == "current":
                as_player = self.current_player
            else:
                raise ValueError("Invalid player selector")
        return {
            "wall": len(self.wall),
            "dealer": self.dealer,
            "round": self.round,
            "status": self.status,
            "current_player": self.current_player,
            "acting_player": self.acting_player,
            "players": [p.to_dict(hide_hand=(as_player is not None and i != as_player)) 
                        for i, p in enumerate(self.players)]
        }

    def to_numpy(self, as_player: int | None = None) -> np.ndarray:
        """
        Turn the game object into a numpy array
        Different from to_dict, this method requires to set up a player as main to mask opponents
        If the player is not given, acting player is used as the main player
        """
        if as_player is None:
            as_player = self.acting_player
        return game_dict_to_numpy(self.to_dict(), player=as_player)
