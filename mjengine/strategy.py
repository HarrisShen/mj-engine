from abc import ABC, abstractmethod
from collections import Counter
import random
from mjengine.constants import PlayerAction
from mjengine.utils import can_chow, can_kong, can_pong, distance_to_ready, is_winning


class Strategy(ABC):
    def __call__(self, hand, discard=False, tile=None, options=None):
        # discard mode
        if discard:
            return self.discard(hand)
        
        # examine hand
        if tile is None:
            if is_winning(hand):
                return self.win(hand)
            if can_kong(hand):
                return self.kong(hand)
            return PlayerAction.PASS, 0
            
        # consider acquiring
        if options is None:
            raise ValueError("options must be specified when considering acquiring")
        if options[3] and is_winning(hand + [tile]):
            return self.win(hand, tile)
        if options[2] and can_kong(hand, tile):
            return self.kong(hand, tile)
        if options[1] and can_pong(hand, tile):
            return self.pong(hand, tile)
        if options[0]:
            return self.chow(hand, tile)
        return PlayerAction.PASS, 0
    
    @abstractmethod
    def discard(self, hand):
        pass

    def win(self, hand, tile=None):
        if tile is None:
            tile = hand[-1]
        return PlayerAction.WIN, tile

    @abstractmethod
    def kong(self, hand, tile=None):
        pass

    @abstractmethod
    def pong(self, hand, tile):
        pass

    @abstractmethod
    def chow(self, hand, tile):
        pass


class RandomStrategy(Strategy):
    
    def discard(self, hand):    
        index = random.randrange(len(hand))
        return index, hand[index]
    
    def kong(self, hand, tile=None):
        if tile is None:
            tile = Counter(hand).most_common(1)[0][0]
        return PlayerAction.KONG, tile
    
    def pong(self, hand, tile):
        return PlayerAction.PONG, tile
    
    def chow(self, hand, tile):
        chow_test = can_chow(hand, tile)
        if chow_test[0]:
            return random.choice([i for i in range(1, 4) if chow_test[i]]), tile
        return PlayerAction.PASS, 0


class ClosestReadyStrategy(Strategy):
    def __init__(self, tiebreak=None) -> None:
        super().__init__()
        self.tiebreak = tiebreak

    def discard(self, hand):
        dist_list = []
        lowest_dist = 14
        for i in range(len(hand)):
            dist_list.append(distance_to_ready(hand[:i] + hand[i + 1:]))
            lowest_dist = min(lowest_dist, dist_list[-1])
        indices = [i for i, d in enumerate(dist_list) if d == lowest_dist]
        if self.tiebreak is None or self.tiebreak == "random":
            index = random.choice(indices)
            return index, hand[index]
        elif self.tiebreak == "value":
            values = [tile_value(hand, hand[i]) for i in indices]
            indices = [i for i, v in zip(indices, values) if v == min(values)]
            index = random.choice(indices)
            return index, hand[index]
    
    def kong(self, hand, tile=None):
        if tile is None:
            candidates = [t for t in hand if hand.count(t) == 4]
        else:
            candidates = [tile]
        for tile in candidates:
            post_hand = [t for t in hand if t != tile]
            if distance_to_ready(post_hand) <= distance_to_ready(hand):
                return PlayerAction.KONG, tile
        return PlayerAction.PASS, 0
    
    def pong(self, hand, tile):
        post_hand = hand.copy()
        post_hand.remove(tile)
        post_hand.remove(tile)
        if distance_to_ready(post_hand) <= distance_to_ready(hand):
            return PlayerAction.PONG, tile
        return PlayerAction.PASS, 0
    
    def chow(self, hand, tile):
        chow_test = can_chow(hand, tile)
        if chow_test[0]:
            distances = [distance_to_ready(hand), 14, 14, 14]
            if chow_test[1]:
                post_hand = hand.copy()
                post_hand.remove(tile - 2)
                post_hand.remove(tile - 1)
                distances[1] = distance_to_ready(post_hand)
            if chow_test[2]:
                post_hand = hand.copy()
                post_hand.remove(tile - 1)
                post_hand.remove(tile + 1)
                distances[2] = distance_to_ready(post_hand)
            if chow_test[3]:
                post_hand = hand.copy()
                post_hand.remove(tile + 1)
                post_hand.remove(tile + 2)
                distances[3] = distance_to_ready(post_hand)
            best_dist = min(distances[1:])
            decision = PlayerAction.PASS
            if best_dist <= distance_to_ready(hand):
                decision = random.choice([i for i, d in enumerate(distances) if i and d == best_dist])
            return decision, tile
        return PlayerAction.PASS, 0
        

def tile_value(hand: list[int], tile: int):
    val = 0
    if hand.count(tile) >= 3:
        val += 6
    elif hand.count(tile) == 2:
        val += 4
    tile_set = set(hand)
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
