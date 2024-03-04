# Mahjong Engine - Models

## Data (game) encoding

Currently, game is encoded using encoding method version 0.1.1.

## Appendix

### 1. Game encoding specifications

Unless specified otherwise, the base unit of sizes specified below is 32bit integer (numpy.int32).

#### Version 0.1.1

Total size: 383

Composition (description: size, range):
- self player's hand: 34, 0-4
- self player's exposed tiles: 34, 0-4
- self player's discarded tiles: 34, 0-4 (counts, no sequence, same for other players')
- other players' exposed and discarded tiles: 3 * 2 * 34 = 204, 0.4
- number of tiles left in the wall: 1, 0-73 (may vary under different rules)
- self player's option: 76, 0-1

#### Version 0.1.0

Total size: 314

Composition (description: size, range):
- self player's hand: 34, 0-4
- self player's exposed tiles: 34, 0-4
- self player's discarded tiles: 33, 0-34 (sequence, same for other players')
- other players' exposed tiles: 3 * 34, 0-4
- other players' discarded tiles: 3 * 33, 0-34
- number of tiles left in the wall: 1, 0-73
- dealer: 1, 0-4
- current player: 1, 0-4
- acting player: 1, 0-4
- (missing fields???)