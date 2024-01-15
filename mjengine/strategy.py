import random
from abc import ABC, abstractmethod

from mjengine.constants import PlayerAction
from mjengine.option import Option
from mjengine.tiles import hand_to_tiles
from mjengine.utils import distance_to_ready_old


class Strategy(ABC):
    def __call__(
            self, hand, discard=False, tile=None, 
            option: Option | None = None) -> tuple[PlayerAction, int]:
        # discard mode
        if discard:
            return self.discard(hand)
        
        # examine hand
        if tile is None:
            if option is None:
                raise ValueError("options must be specified when examining hand")
            if option.win_from_self:
                action, win_tile = self.win(hand)
                if action == PlayerAction.WIN:
                    return action, win_tile
            if option.concealed_kong:
                action, kong_tile = self.kong(hand, option.concealed_kong)
                if action == PlayerAction.KONG:
                    return action, kong_tile
            return PlayerAction.PASS, 0
            
        # consider acquiring
        if option is None:
            raise ValueError("options must be specified when considering acquiring")
        if option.win_from_chuck:
            action, _ = self.win(hand, tile)
            if action == PlayerAction.WIN:
                return action, tile
        if option.exposed_kong:
            action, _ = self.kong(hand, [tile])
            if action == PlayerAction.KONG:
                return action, tile
        if option.pong:
            action, _ = self.pong(hand, tile)
            if action == PlayerAction.PONG:
                return action, tile
        if option.chow[0]:
            action, _ = self.chow(hand, tile, option=option.chow)
            if action != PlayerAction.PASS:
                return action, tile
        return PlayerAction.PASS, 0
    
    @abstractmethod
    def discard(self, hand: list[int]):
        pass

    def win(self, hand: list[int], tile=None):
        return PlayerAction.WIN, tile

    @abstractmethod
    def kong(self, hand: list[int], tiles=None):
        pass

    @abstractmethod
    def pong(self, hand: list[int], tile):
        pass

    @abstractmethod
    def chow(self, hand: list[int], tile, option):
        pass


class RandomStrategy(Strategy):
    
    def discard(self, hand):
        return random.choice(hand_to_tiles(hand))
    
    def kong(self, hand, tiles=None):
        if tiles is None:
            tiles = [t for t in range(len(hand)) if hand[t] == 4]
        return PlayerAction.KONG, random.choice(tiles)
    
    def pong(self, hand, tile):
        return PlayerAction.PONG, tile
    
    def chow(self, hand, tile, option):
        return random.choice([i for i in range(1, 4) if option[i]]), tile


class ClosestReadyStrategy(Strategy):
    def __init__(self, tiebreak=None) -> None:
        super().__init__()
        self.tiebreak = tiebreak

    def discard(self, hand):
        dist_list = [14 for _ in range(len(hand))]
        lowest_dist = 14
        for i in range(len(hand)):
            if hand[i] == 0:
                continue
            new_hand = hand.copy()
            new_hand[i] -= 1
            dist_list[i] = distance_to_ready_old(new_hand)
            lowest_dist = min(lowest_dist, dist_list[-1])
        best_tiles = [i for i, d in enumerate(dist_list) if d == lowest_dist]
        if self.tiebreak == "value":
            values = [tile_value(hand, tile) for tile in best_tiles]
            best_value = min(values)
            best_tiles = [tile for tile, value in zip(best_tiles, values) if value == best_value]
        return random.choice(best_tiles)
    
    def kong(self, hand, tiles=None):
        if tiles is None:
            tiles = [i for i in range(len(hand)) if hand[i] == 4]
        for tile in tiles:
            new_hand = hand.copy()
            new_hand[tile] = 0
            if distance_to_ready_old(new_hand) <= distance_to_ready_old(hand):
                return PlayerAction.KONG, tile
        return PlayerAction.PASS, -1
    
    def pong(self, hand, tile):
        new_hand = hand.copy()
        new_hand[tile] -= 2
        if distance_to_ready_old(new_hand) <= distance_to_ready_old(hand):
            return PlayerAction.PONG, tile
        return PlayerAction.PASS, 0
    
    def chow(self, hand, tile, option):
        distances = [distance_to_ready_old(hand), 14, 14, 14]
        if option[1]:
            new_hand = hand.copy()
            new_hand[tile - 2] -= 1
            new_hand[tile - 1] -= 1
            distances[1] = distance_to_ready_old(new_hand)
        if option[2]:
            new_hand = hand.copy()
            new_hand[tile - 1] -= 1
            new_hand[tile + 1] -= 1
            distances[2] = distance_to_ready_old(new_hand)
        if option[3]:
            new_hand = hand.copy()
            new_hand[tile + 1] -= 1
            new_hand[tile + 2] -= 1
            distances[3] = distance_to_ready_old(new_hand)
        best_dist = min(distances[1:])
        decision = PlayerAction.PASS
        if best_dist <= distance_to_ready_old(hand):
            decision = random.choice([i for i, d in enumerate(distances) if i and d == best_dist])
        return decision, tile
        

def tile_value(hand: list[int], tile: int):
    val = 0
    if hand[tile] >= 3:
        val += 6
    elif hand[tile] == 2:
        val += 4
    tile_set = set(i for i in range(len(hand)) if hand[i])
    if tile - 2 in tile_set and tile - 1 in tile_set:
        val += 6
    elif tile - 1 in tile_set and tile + 1 in tile_set:
        val += 6
    elif tile + 1 in tile_set and tile + 2 in tile_set:
        val += 6
    if tile < 40:
        suit, num = tile // 10, tile % 10
        if 4 <= num <= 6:
            val += 3
        elif num == 3 or num == 7:
            val += 2
        elif num == 2 or num == 8:
            val += 1
        nearest = 10
        for t in range(suit * 10 + 1, suit * 10 + 10):
            if t == tile:
                continue
            if t in tile_set:
                nearest = min(nearest, abs(t - tile))
        if nearest == 1:
            val += 2
        elif nearest == 2:
            val += 1
    return val
