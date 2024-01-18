from mjengine.game import Game
from mjengine.player import make_player


if __name__ == "__main__":
    players = [make_player("random") for _ in range(3)]
    model_dir = "D:\\Coding\\mj-project\\mj-engine\\trained_models\\DQN_128_default_240118203226"
    players.append(make_player(model_dir))
    game = Game(players=players, round_limit=1024, game_limit=None, seed=None, verbose=0)
    game.play()
    print("Game finished.")
    print(game.score_summary(detail=1))
