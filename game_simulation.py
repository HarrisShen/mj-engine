from mjengine.game import Game
from mjengine.player import make_player


if __name__ == "__main__":
    players = [make_player("closest") for _ in range(3)]
    players.append(make_player("closest_value"))
    game = Game(players=players, round_limit=4, game_limit=None, verbose=True)
    game.play()
    print("Game finished.")
    game.score_summary()
