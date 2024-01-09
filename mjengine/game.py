from collections import deque
import random
from mjengine.constants import TID_LIST, GameStatus, PlayerAction
from mjengine.option import Option
from mjengine.player import Player
from mjengine.utils import can_chow, can_kong, can_pong, is_winning, tid_to_unicode


class Game:
    def __init__(
            self, 
            players: list[Player] | None = None, 
            round_limit: int = 1) -> None:
        self.round_limit = round_limit

        self.wall = []
        self.dealer = 0
        self.round = 0
        self.status = GameStatus.START
        if players is None:
            players = [Player() for _ in range(4)]
        self.players = players
        self.current_player = 0
        self.decisions = [None for _ in range(4)]
        self.options = []
        self.acting_queue = None
        self.waiting = []

    def reset(self) -> None:
        self.wall = []
        self.status = GameStatus.START
        for i in range(4):
            self.players[i].reset()

    def start_game(self) -> None:
        if self.status != GameStatus.START:
            raise ValueError("Game is not ready to deal")
        
        self.wall = TID_LIST * 4
        random.shuffle(self.wall)

        # mimic real procedure - deal 4 tiles to each player, then 1 tile to each player
        for _ in range(3):
            for i in range(4):
                self.deal((i + self.dealer) % 4, 4)
        for i in range(4):
            self.deal((i + self.dealer) % 4, 1)
            print(f"Player {(i + self.dealer) % 4}:", self.players[(i + self.dealer) % 4].hand_to_str())

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
        while self.round < self.round_limit:
            self.reset()
            print(f"Round {self.round} - Dealer is Player {self.dealer} now.")
            self.start_game()
            while self.status < GameStatus.END:
                player, option, last_discard = self.get_option()
                action, tile = self.players[player].decide(option, last_discard)
                self.apply_action(action, player, tile)
            self.settle_score()
            if not self.players[self.dealer].is_winning():
                self.dealer = (self.dealer + 1) % 4
                # end of a round
                if self.dealer == 0:
                    self.round += 1

    def get_option(self) -> tuple[int, Option, int | None]:
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
            print(f"Player {self.current_player} draws {tid_to_unicode(last_draw)}")
            print(f"Player {self.current_player}:", self.players[self.current_player].hand_to_str())
            hand = self.players[self.current_player].hand
            kong_tiles = [tid for tid in set(hand) if hand.count(tid) == 4]
            self_win = is_winning(hand)
            if self_win or kong_tiles:
                option = Option(concealed_kong=kong_tiles, win_from_self=self_win)
                return self.current_player, option, None
            self.status = GameStatus.DISCARD

        # discard a tile
        if self.status == GameStatus.DISCARD:
            option = Option(discard=True)
            return self.current_player, option, None

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
                    exposed_kong=can_kong(hand, last_discard),
                    win_from_chuck=is_winning(hand + [last_discard])
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
            return self.get_option()
        pid, option = self.acting_queue.popleft()
        return pid, option, last_discard

    def apply_action(self, action: PlayerAction, player: int, tile: int | None) -> None:
        if action is None:
            if self.status != GameStatus.DISCARD:
                raise ValueError("Invalid action")
            self.players[player].discard(tile)
            print(f"Player {player} discards {tid_to_unicode(tile)}")
            print(f"Player {player}:", self.players[player].hand_to_str())
            if not self.wall:
                print("Wall is empty!")
                self.status = GameStatus.END
                return
            self.status = GameStatus.CHECK
            return
        
        hand = self.players[player].hand
        if action == PlayerAction.WIN:
            if not is_winning(hand) and not is_winning(hand + [tile]):
                raise ValueError("Invalid action")
        elif action == PlayerAction.KONG:
            if not can_kong(self.players[player].hand, tile):
                raise ValueError("Invalid action")
        elif action == PlayerAction.PONG:
            if not can_pong(self.players[player].hand, tile):
                raise ValueError("Invalid action")
        elif action > PlayerAction.PASS:  # chow
            if not can_chow(self.players[player].hand, tile)[0]:
                raise ValueError("Invalid action")
        self.waiting.append((player, action))

        # check whether the action can be taken right away
        if self.acting_queue and action <= self.acting_queue[0][1].highest_action():
            # need to wait
            return
        # apply the action
        self.acting_queue = None # reset acting queue
        self.waiting.sort(key=lambda x: x[1], reverse=True)
        action = self.waiting[0][1]
        acting_players = [i for i, a in self.waiting if a == action]
        self.waiting = []

        if action == PlayerAction.WIN:
            for i in acting_players:
                self.players[i].won = True
                print(f"Player {i} wins!")
                print(f"Player {i}:", self.players[i].hand_to_str())
            self.status = GameStatus.END
            return
        
        if action == PlayerAction.KONG:
            self.players[acting_players[0]].kong(tile)
            print(f"Player {acting_players[0]} kongs!")
            self.current_player = acting_players[0]
            self.status = GameStatus.DRAW
            return
        
        if action == PlayerAction.PONG:
            self.players[acting_players[0]].pong(tile)
            print(f"Player {acting_players[0]} pongs!")
            self.current_player = acting_players[0]
            self.status = GameStatus.DISCARD
            return
        
        if action > PlayerAction.PASS:  # chow
            self.players[acting_players[0]].chow(tile, action)
            print(f"Player {acting_players[0]} chows!")
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

    def score_summary(self) -> None:
        """Print the score summary of the game in a table format.
        """
        print("P\tScore\tW\tSW\tC")
        for i in range(4):
            print(f"{i}\t{self.players[i].score}\t{self.players[i].wins}\t"
                  f"{self.players[i].self_wins}\t{self.players[i].chucks}")
            
    def to_dict(self, as_player: int | None = None) -> dict:
        """Return a dict representation of the game.
        """
        return {
            "wall": len(self.wall),
            "dealer": self.dealer,
            "round": self.round,
            "status": self.status,
            "active_player": self.current_player,
            "players": [p.to_dict(hide_hand=(as_player is not None and i != as_player)) 
                        for i, p in enumerate(self.players)]
        }
    