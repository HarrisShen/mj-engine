import random
from mjengine.constants import GameStatus, PlayerAction
from mjengine.player import Player
from mjengine.utils import TID_LIST, tid_to_unicode


class Game:
    def __init__(self, players: list[Player] | None = None, round_limit: int = 1) -> None:
        self.round_limit = round_limit

        self.wall = []
        self.dealer = 0
        self.round = 0
        self.status = GameStatus.START
        if players is None:
            players = [Player() for _ in range(4)]
        self.players = players
        self.active_player = 0

    def reset(self) -> None:
        self.wall = []
        self.status = GameStatus.START
        for i in range(4):
            self.players[i].reset()

    def deal(self) -> None:
        if self.status != GameStatus.START:
            raise ValueError("Game is not ready to deal")
        
        self.wall = TID_LIST * 4
        random.shuffle(self.wall)

        # mimic real procedure - deal 4 tiles to each player, then 1 tile to each player
        for _ in range(3):
            for i in range(4):
                self.players[(i + self.dealer) % 4].draw(self.wall, 4)
        for i in range(4):
            self.players[(i + self.dealer) % 4].draw(self.wall, 1)
            print(f"Player {(i + self.dealer) % 4}:", self.players[(i + self.dealer) % 4].hand_to_str())
        # self.players[self.dealer].draw(self.wall, 1)
        # print(f"Player {self.dealer}: ", self.players[self.dealer].hand_to_str())
        self.active_player = self.dealer
        self.status = GameStatus.DRAW

    def play(self) -> None:
        while self.round < self.round_limit:
            self.reset()
            self.deal()
            while self.status < GameStatus.END:
                self.next_action()
            self.settle_score()
            if not self.players[self.dealer].is_winning():
                self.dealer = (self.dealer + 1) % 4
                print(f"Dealer is Player {self.dealer} now.")
                # end of a round
                if self.dealer == 0:
                    self.round += 1

    def next_action(self, action=None, tile=None) -> None:
        if self.status == GameStatus.START:
            raise ValueError("Game is not ready to play, please deal first")
        
        # draw a tile and check winning
        if self.status == GameStatus.DRAW:
            last_draw = self.players[self.active_player].draw(self.wall, 1)[0]
            print(f"Player {self.active_player} draws {tid_to_unicode(last_draw)}")
            print(f"Player {self.active_player}:", self.players[self.active_player].hand_to_str())
            action, tile = self.players[self.active_player].examine()
            # win by self-draw
            if action == PlayerAction.WIN:
                self.players[self.active_player].won = True  # award the win
                print(f"Player {self.active_player} wins!")
                print(f"Player {self.active_player}:", self.players[self.active_player].hand_to_str())
                self.status = GameStatus.END
                return
            # kong
            if action == PlayerAction.KONG:
                self.players[self.active_player].kong(tile)
                print(f"Player {self.active_player} kongs!")
                self.status = GameStatus.DRAW
                return
            self.status = GameStatus.DISCARD

        # discard a tile
        elif self.status == GameStatus.DISCARD:
            tile = self.players[self.active_player].discard()
            print(f"Player {self.active_player} discards {tid_to_unicode(tile)}")
            print(f"Player {self.active_player}:", self.players[self.active_player].hand_to_str())
            if not self.wall:
                print("Wall is empty!")
                self.status = GameStatus.END
                return
            self.status = GameStatus.CHECK

        # check chow, pong, kong and chuck (winning from others' discards)
        elif self.status == GameStatus.CHECK:
            last_discard = self.players[self.active_player].discards[-1]
            decisions = [PlayerAction.PASS for _ in range(4)]
            for i in range(4):
                if i == self.active_player:
                    continue
                # decide whether options are available - chow, pong, kong, chuck
                options = [i == (self.active_player + 1) % 4, True, True, True]
                decisions[i] = self.players[i].acquire(last_discard, options)
            # fulfill decisions based on priority - win by chuck > kong > pong > chow
            action = max(decisions)
            acting_players = [i for i, d in enumerate(decisions) if d == action]
            # allow chuck multiple players for one tile
            # this is the only case where multiple players can act at the same time
            if action == PlayerAction.WIN:
                for i in acting_players:
                    self.players[i].won = True  # award the win
                    print(f"Player {i} wins!")
                    print(f"Player {i}:", self.players[i].hand_to_str())
                self.status = GameStatus.END
                return
            # apply chow, pong, kong
            if action == PlayerAction.KONG:
                self.players[acting_players[0]].kong(last_discard)
                print(f"Player {acting_players[0]} kongs!")
                self.active_player = acting_players[0]
                self.status = GameStatus.DRAW
            elif action == PlayerAction.PONG:
                self.players[acting_players[0]].pong(last_discard)
                print(f"Player {acting_players[0]} pongs!")
                self.active_player = acting_players[0]
                self.status = GameStatus.DISCARD
            elif action > PlayerAction.PASS:
                self.players[acting_players[0]].chow(last_discard, action)
                print(f"Player {acting_players[0]} chows!")
                self.active_player = acting_players[0]
                self.status = GameStatus.DISCARD
            # all pass
            else:
                self.active_player = (self.active_player + 1) % 4
                self.status = GameStatus.DRAW
    
    def settle_score(self) -> None:
        # no one wins
        if all([not self.players[i].won for i in range(4)]):
            return
        
        # win by self-draw
        if self.players[self.active_player].won:
            self.players[self.active_player].wins += 1
            self.players[self.active_player].self_wins += 1
            for i in range(4):
                if i == self.active_player:
                    continue
                self.players[i].score -= 1
            self.players[self.active_player].score += 3
            return
        
        # win by chuck
        self.players[self.active_player].chucks += 1
        for i in range(4):
            if self.players[i].won:
                self.players[i].wins += 1
                self.players[i].score += 1
                self.players[self.active_player].score -= 1

    def score_summary(self) -> None:
        """Print the score summary of the game in a table format.
        """
        print("P\tScore\tW\tSW\tC")
        for i in range(4):
            print(f"{i}\t{self.players[i].score}\t{self.players[i].wins}\t"
                  f"{self.players[i].self_wins}\t{self.players[i].chucks}")
    