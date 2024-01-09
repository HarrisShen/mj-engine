from dataclasses import dataclass, field

from mjengine.constants import PlayerAction


@dataclass
class Option:
    discard: bool = False
    concealed_kong: list[int] = field(default_factory=list)
    win_from_self: bool = False
    chow: list[bool] = field(default_factory=lambda: [False, False, False, False])
    pong: bool = False
    exposed_kong: bool = False
    win_from_chuck: bool = False

    def tier(self) -> tuple[int, int]:
        """
        Helper method to determine the priority of options.
        Note that only when considering tile from others', the tier is meaningful.
        In multiple winners situation, the winner having only one option is prioritized.
        """
        if self.win_from_chuck:
            if self.exposed_kong or self.pong:
                return 3, 2
            if self.chow[0]:
                return 3, 1
            return 3, 3
        if self.exposed_kong or self.pong:
            return 2, 0
        if self.chow[0]:
            return 1, 0
        return 0, 0
    
    def highest_action(self) -> PlayerAction:
        if self.win_from_chuck:
            return PlayerAction.WIN
        if self.exposed_kong:
            return PlayerAction.KONG
        if self.pong:
            return PlayerAction.PONG
        if self.chow[0]:
            return PlayerAction.CHOW3
        return PlayerAction.PASS
