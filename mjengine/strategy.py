import random
from abc import ABC, abstractmethod

import numpy as np

from mjengine.constants import PlayerAction
from mjengine.models.agents import Agent, DQN
from mjengine.models.utils import game_dict_to_numpy, parse_action
from mjengine.option import Option
from mjengine.shanten import Shanten
from mjengine.tiles import hand_to_tiles


class Strategy(ABC):
    def __call__(
            self,
            hand: list[int],
            info: dict,
            discard: bool = False,
            tile: int | None = None,
            option: Option | None = None) -> tuple[PlayerAction, int]:
        # discard mode
        if discard:
            return self.discard(hand, info)
        
        # examine hand
        if tile is None:
            if option is None:
                raise ValueError("options must be specified when examining hand")
            if option.win_from_self:
                action, win_tile = self.win(hand, info)
                if action == PlayerAction.WIN:
                    return action, win_tile
            if option.concealed_kong:
                action, kong_tile = self.kong(hand, info, option.concealed_kong)
                if action == PlayerAction.KONG:
                    return action, kong_tile
            return PlayerAction.PASS, 0
            
        # consider acquiring
        if option is None:
            raise ValueError("options must be specified when considering acquiring")
        if option.win_from_chuck:
            action, _ = self.win(hand, info, tile)
            if action == PlayerAction.WIN:
                return action, tile
        if option.exposed_kong:
            action, _ = self.kong(hand, info, [tile])
            if action == PlayerAction.KONG:
                return action, tile
        if option.pong:
            action, _ = self.pong(hand, info, tile)
            if action == PlayerAction.PONG:
                return action, tile
        if option.chow[0]:
            action, _ = self.chow(hand, info, tile, option.chow)
            if action != PlayerAction.PASS:
                return action, tile
        return PlayerAction.PASS, 0
    
    @abstractmethod
    def discard(self, hand: list[int], info: dict):
        pass

    def win(self, hand: list[int], info: dict, tile: int | None = None):
        return PlayerAction.WIN, tile

    @abstractmethod
    def kong(self, hand: list[int], info: dict, tiles: list[int] | None):
        pass

    @abstractmethod
    def pong(self, hand: list[int], info: dict, tile: int):
        pass

    @abstractmethod
    def chow(self, hand: list[int], info: dict, tile: int, option: list[bool]):
        pass


class RandomStrategy(Strategy):
    
    def discard(self, hand, info):
        return None, random.choice(hand_to_tiles(hand))
    
    def kong(self, hand, info, tiles=None):
        if tiles is None:
            tiles = [t for t in range(len(hand)) if hand[t] == 4]
        return PlayerAction.KONG, random.choice(tiles)
    
    def pong(self, hand, info, tile):
        return PlayerAction.PONG, tile
    
    def chow(self, hand, info, tile, option):
        return random.choice([i for i in range(1, 4) if option[i]]), tile


class ClosestReadyStrategy(Strategy):
    def __init__(self, tiebreak=None, index_dir=".") -> None:
        super().__init__()
        self.tiebreak = tiebreak
        self.shanten = Shanten()
        self.shanten.prepare(index_dir)

    def discard(self, hand, info):
        dist_list = [14 for _ in range(len(hand))]
        for i in range(len(hand)):
            if hand[i] == 0:
                continue
            hand[i] -= 1
            dist_list[i] = self.shanten(hand)
            hand[i] += 1
        lowest_dist = min(dist_list)
        best_tiles = [tid for tid, d in enumerate(dist_list) if d == lowest_dist]
        if self.tiebreak == "value":
            values = [tile_value(hand, tile) for tile in best_tiles]
            best_value = min(values)
            best_tiles = [tile for tile, value in zip(best_tiles, values) if value == best_value]
        return None, random.choice(best_tiles)
    
    def kong(self, hand, info, tiles=None):
        if tiles is None:
            tiles = [i for i in range(len(hand)) if hand[i] == 4]
        for tile in tiles:
            new_hand = hand.copy()
            new_hand[tile] = 0
            if self.shanten(new_hand) <= self.shanten(hand):
                return PlayerAction.KONG, tile
        return PlayerAction.PASS, -1
    
    def pong(self, hand, info, tile):
        new_hand = hand.copy()
        new_hand[tile] -= 2
        if self.shanten(new_hand) <= self.shanten(hand):
            return PlayerAction.PONG, tile
        return PlayerAction.PASS, 0
    
    def chow(self, hand, info, tile, option):
        distances = [self.shanten(hand), 14, 14, 14]
        if option[1]:
            new_hand = hand.copy()
            new_hand[tile - 2] -= 1
            new_hand[tile - 1] -= 1
            distances[1] = self.shanten(new_hand)
        if option[2]:
            new_hand = hand.copy()
            new_hand[tile - 1] -= 1
            new_hand[tile + 1] -= 1
            distances[2] = self.shanten(new_hand)
        if option[3]:
            new_hand = hand.copy()
            new_hand[tile + 1] -= 1
            new_hand[tile + 2] -= 1
            distances[3] = self.shanten(new_hand)
        best_dist = min(distances[1:])
        decision = PlayerAction.PASS
        if best_dist <= self.shanten(hand):
            decision = random.choice([i for i, d in enumerate(distances) if i and d == best_dist])
        return decision, tile
        

def tile_value(hand: list[int], tile: int):
    val = 0
    if hand[tile] >= 3:
        val += 6
    elif hand[tile] == 2:
        val += 4
    if tile < 27:
        check_val = {i: (tile + i) % 9 == tile % 9 and hand[tile + i] for i in range(-2, 3)}
        if check_val[-2] and check_val[-1]:
            val += 6
        elif check_val[-1] and check_val[1]:
            val += 6
        elif check_val[1] and check_val[2]:
            val += 6
        suit, num = tile // 9, tile % 9 + 1
        if 4 <= num <= 6:
            val += 3
        elif num == 3 or num == 7:
            val += 2
        elif num == 2 or num == 8:
            val += 1
        nearest = 10
        for t in range(suit * 9, suit * 9 + 9):
            if t == tile:
                continue
            if hand[t]:
                nearest = min(nearest, abs(t - tile))
        if nearest == 1:
            val += 2
        elif nearest == 2:
            val += 1
    return val


class RLAgentStrategy(Strategy):
    def __init__(self, agent: Agent):
        self.agent = agent

    def discard(self, hand, info):
        state = game_dict_to_numpy(info)
        option = np.zeros(76, dtype=bool)
        for i in range(34):
            option[i] = hand[i] > 0
        action = self.agent.take_action(state, option)
        action, tile = parse_action(action)
        return action, tile

    def win(self, hand: list[int], info: dict, tile: int | None = None):
        state = game_dict_to_numpy(info)
        option = np.zeros(76, dtype=bool)
        if tile is None:
            option[68] = True
        else:
            option[74] = True
        option[75] = True
        action = self.agent.take_action(state, option)
        action, _ = parse_action(action)
        return action, tile

    def kong(self, hand: list[int], info: dict, tiles: list[int] | None):
        state = game_dict_to_numpy(info)
        option = np.zeros(76, dtype=bool)
        if hand[tiles[0]] == 4:  # concealed kong
            for tid in tiles:
                option[tid + 34] = True
        else:
            option[73] = True
        option[75] = True
        action = self.agent.take_action(state, option)
        action, tile = parse_action(action)
        return action, tile if tile is not None else tiles[0]

    def pong(self, hand: list[int], info: dict, tile: int):
        state = game_dict_to_numpy(info)
        option = np.zeros(76, dtype=bool)
        option[72] = True
        option[75] = True
        action = self.agent.take_action(state, option)
        action, _ = parse_action(action)
        return action, tile

    def chow(self, hand: list[int], info: dict, tile: int, option: list[bool]):
        state = game_dict_to_numpy(info)
        option = np.zeros(76, dtype=bool)
        for i in range(1, 4):
            option[68 + i] = option[i]
        option[75] = True
        action = self.agent.take_action(state, option)
        action, _ = parse_action(action)
        return action, tile

    @staticmethod
    def load(model_dir, device):
        agent = DQN.restore(model_dir, device)
        return RLAgentStrategy(agent)
