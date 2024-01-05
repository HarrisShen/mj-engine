import random
from mjengine.constants import GameStatus, PlayerAction
from mjengine.player import Player
from mjengine.utils import TID_LIST, can_chow, can_kong, can_pong, is_winning, tid_to_unicode


class Game:
    def __init__(self, players: list[Player] | None = None, round_limit: int = 1) -> None:
        self.round_limit = round_limit

        self.wall = []
        self.dealer = 0
        self.round = 0
        self.status = GameStatus.START
        if players is None:
            players = [Player() for _ in range(4)]
        self.register(players)
        self.active_player = 0

    def register(self, players: list[Player]) -> None:
        if len(players) != 4:
            raise ValueError("Game requires 4 players")
        self.players = players
        for i in range(4):
            self.players[i].game = self
            self.players[i].position = i

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
                tiles = self.draw(4)
                self.players[(i + self.dealer) % 4].draw(tiles)
        for i in range(4):
            tiles = self.draw(1)
            self.players[(i + self.dealer) % 4].draw(tiles)
            print(f"Player {(i + self.dealer) % 4}:", self.players[(i + self.dealer) % 4].hand_to_str())

        self.active_player = self.dealer
        self.status = GameStatus.DRAW

    def draw(self, n: int = 1) -> list[int]:
        """Draw from the wall"""
        if n > len(self.wall):
            raise ValueError("Insufficient tiles in the wall")
        tiles = self.wall[-n:]
        self.wall = self.wall[:-n]
        return tiles

    def play(self) -> None:
        while self.round < self.round_limit:
            self.reset()
            self.deal()
            while self.status < GameStatus.END:
                action, player, tile = self.get_next_action()
                self.apply_action(action, player, tile)
            self.settle_score()
            if not self.players[self.dealer].is_winning():
                self.dealer = (self.dealer + 1) % 4
                print(f"Dealer is Player {self.dealer} now.")
                # end of a round
                if self.dealer == 0:
                    self.round += 1

    def get_next_action(self) -> tuple[PlayerAction, list[int], int]:
        if self.status == GameStatus.START:
            raise ValueError("Game is not ready to play, please deal first")
        
        # draw a tile and check winning
        if self.status == GameStatus.DRAW:
            last_draw = self.draw(1)[0]
            self.players[self.active_player].draw([last_draw])
            print(f"Player {self.active_player} draws {tid_to_unicode(last_draw)}")
            print(f"Player {self.active_player}:", self.players[self.active_player].hand_to_str())
            action, tile = self.players[self.active_player].examine()
            return action, [self.active_player], tile

        # discard a tile
        if self.status == GameStatus.DISCARD:
            tile = self.players[self.active_player].select_discard()
            return None, [self.active_player], tile

        # check chow, pong, kong and chuck (winning from others' discards)
        last_discard = self.players[self.active_player].discards[-1]
        decisions = [PlayerAction.PASS for _ in range(4)]
        for i in range(4):
            if i == self.active_player:
                continue
            # decide whether options are available - chow, pong, kong, chuck
            hand = self.players[i].hand
            options = [
                i == (self.active_player + 1) % 4 and can_chow(hand, last_discard), 
                can_pong(hand, last_discard),
                can_kong(hand, last_discard),
                is_winning(hand + [last_discard])
            ]
            if not any(options):
                continue
            decisions[i] = self.players[i].acquire(last_discard, options)
        # fulfill decisions based on priority - win by chuck > kong > pong > chow
        action = max(decisions)
        acting_players = [i for i, d in enumerate(decisions) if d == action]
        return action, acting_players, last_discard
    
    def apply_action(self, action: PlayerAction, players: list[int], tile: int | None) -> None:
        if action is None:
            self.players[players[0]].discard(tile)
            print(f"Player {players[0]} discards {tid_to_unicode(tile)}")
            print(f"Player {players[0]}:", self.players[players[0]].hand_to_str())
            if not self.wall:
                print("Wall is empty!")
                self.status = GameStatus.END
                return
            self.status = GameStatus.CHECK
            return
        
        if action == PlayerAction.WIN:
            for i in players:
                self.players[i].won = True
                print(f"Player {i} wins!")
                print(f"Player {i}:", self.players[i].hand_to_str())
            self.status = GameStatus.END
            return
        
        # all actions other than win can only be performed by one player
        if action == PlayerAction.KONG:
            self.players[players[0]].kong(tile)
            print(f"Player {players[0]} kongs!")
            self.active_player = players[0]
            self.status = GameStatus.DRAW
            return
        
        if action == PlayerAction.PONG:
            self.players[players[0]].pong(tile)
            print(f"Player {players[0]} pongs!")
            self.active_player = players[0]
            self.status = GameStatus.DISCARD
            return
        
        if action > PlayerAction.PASS:  # chow
            self.players[players[0]].chow(tile, action)
            print(f"Player {players[0]} chows!")
            self.active_player = players[0]
            self.status = GameStatus.DISCARD
            return
        
        if action == PlayerAction.PASS:
            # all pass after check
            if self.status == GameStatus.CHECK:
                self.active_player = (self.active_player + 1) % 4
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
    