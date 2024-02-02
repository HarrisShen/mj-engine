import os.path
import random
from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np

from mjengine.analyzer import Analyzer
from mjengine.constants import PlayerAction
from mjengine.models.agent.agent import Agent
from mjengine.models.agent.dqn import DQN
from mjengine.models.agent.ppo import PPO
from mjengine.models.utils import parse_action, game_dict_to_numpy
from mjengine.option import Option
from mjengine.tiles import hand_to_tiles

StratOutput: TypeAlias = tuple[PlayerAction | None, int]


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
    def discard(self, hand: list[int], info: dict) -> StratOutput:
        pass

    def win(self, hand: list[int], info: dict, tile: int | None = None) -> StratOutput:
        return PlayerAction.WIN, tile

    @abstractmethod
    def kong(self, hand: list[int], info: dict, tiles: list[int] | None) -> StratOutput:
        pass

    @abstractmethod
    def pong(self, hand: list[int], info: dict, tile: int) -> StratOutput:
        pass

    @abstractmethod
    def chow(self, hand: list[int], info: dict, tile: int, option: list[bool]) -> StratOutput:
        pass


class RandomStrategy(Strategy):
    def __init__(self, level: int):
        super().__init__()

        # Random level:
        # 0 - total random
        # 1 - discard random, always chow/pong/kong
        if level < 0 and level > 1:
            raise ValueError("Invalid level value, expected 0 or 1")
        self.level = level
    
    def discard(self, hand, info):
        return None, random.choice(hand_to_tiles(hand))
    
    def kong(self, hand, info, tiles=None):
        if tiles is None:
            tiles = [t for t in range(len(hand)) if hand[t] == 4]
        if self.level == 0:
            tiles += [-1]
        tile = random.choice(tiles)
        if tile == -1:
            return PlayerAction.PASS, -1
        return PlayerAction.KONG, tile
    
    def pong(self, hand, info, tile):
        if self.level == 0:
            tile = random.choice([-1, tile])
        if tile == -1:
            return PlayerAction.PASS, tile
        return PlayerAction.PONG, tile
    
    def chow(self, hand, info, tile, option):
        action = random.choice([i for i in range(self.level, 4) if option[i]])
        return PlayerAction(action), tile


class AnalyzerStrategy(Strategy):

    def __init__(self, tiebreak=None, index_dir="./index/") -> None:
        super().__init__()
        self.tiebreak = tiebreak
        self.analyzer = Analyzer()
        self.analyzer.prepare(index_dir)

    def discard(self, hand, info):
        _, discard, _ = self.analyzer(hand)
        best_discards = [tid for tid, v in enumerate(discard) if v]
        if self.tiebreak == "value":
            costs = [tile_value(hand, tid) if discard[tid] else 1024 for tid in range(len(hand))]
            best_cost = min(costs)
            best_discards = [tid for tid in best_discards if costs[tid] == best_cost]
        elif self.tiebreak == "exp0" or self.tiebreak == "exp1":
            # exp0: Number of types of expected tiles
            # exp1: Number of tiles of expected tiles, presumed
            best_discards, _ = self.analyzer.best_discard(
                hand, None if self.tiebreak == "exp0" else info)
        return None, random.choice(best_discards)
    
    def kong(self, hand, info, tiles=None):
        if tiles is None:
            tiles = [i for i in range(len(hand)) if hand[i] == 4]
        st, _, wait = self.analyzer(hand)
        for tile in tiles:
            new_hand = hand.copy()
            new_hand[tile] = 0
            st1, _, wait1 = self.analyzer(new_hand)
            if (-st1, sum(wait1)) >= (-st, sum(wait)):
                return PlayerAction.KONG, tile
        return PlayerAction.PASS, -1
    
    def pong(self, hand, info, tile):
        st, _, wait = self.analyzer(hand)
        new_hand = hand.copy()
        new_hand[tile] -= 2
        st1, _, wait1 = self.analyzer(new_hand)
        if (-st1, sum(wait1)) >= (-st, sum(wait)):
            return PlayerAction.PONG, tile
        return PlayerAction.PASS, tile
    
    def chow(self, hand, info, tile, option):
        st, _, wait = self.analyzer(hand)
        sts = [st, 14, 14, 14]
        waits = [sum(wait), 0, 0, 0]
        if option[1]:
            new_hand = hand.copy()
            new_hand[tile - 2] -= 1
            new_hand[tile - 1] -= 1
            sts[1], _, new_wait = self.analyzer(new_hand)
            waits[1] = sum(new_wait)
        if option[2]:
            new_hand = hand.copy()
            new_hand[tile - 1] -= 1
            new_hand[tile + 1] -= 1
            sts[2], _, new_wait = self.analyzer(new_hand)
            waits[2] = sum(new_wait)
        if option[3]:
            new_hand = hand.copy()
            new_hand[tile + 1] -= 1
            new_hand[tile + 2] -= 1
            sts[3], _, new_wait = self.analyzer(new_hand)
            waits[3] = sum(new_wait)
        best_status = (-sts[0], waits[0])
        decision = [0]
        for i in range(1, 4):
            if (-sts[i], waits[i]) > best_status:
                best_status = (-sts[i], waits[i])
                decision = [i]
            elif (-sts[i], waits[i]) == best_status:
                decision.append(i)
        # if best_dist <= self.analyzer(hand):
        #     decision = random.choice([i for i, d in enumerate(distances) if i and d == best_dist])
        decision = PlayerAction(random.choice(decision))
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
        model_name = os.path.split(model_dir)[-1]
        if model_name.startswith("DQN"):
            agent = DQN.restore(model_dir, device, train=False)
        elif model_name.startswith("PPO") or model_name.startswith("GAIL_PPO"):
            agent = PPO.restore(model_dir, device, train=False)
        else:
            raise ValueError("No proper model class detected")
        return RLAgentStrategy(agent)
