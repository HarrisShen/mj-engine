import argparse
import time

from mjengine.game import Game
from mjengine.player import make_player


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("players",
                        help="Player types or model directory if using a trained model", nargs=4)
    parser.add_argument("-r", "--round", help="Round limit of the game", type=int)
    parser.add_argument("-g", "--game", help="Limit of game to be played", type=int)
    parser.add_argument("-s", "--seed", help="Random seed used for game")
    parser.add_argument("-v", "--verbose", help="Verbosity of the program", action="count")
    args = parser.parse_args()

    print("Preparing game...")
    start = time.process_time()
    players = [make_player(args.players[i]) for i in range(4)]
    # model_dir = "D:\\Coding\\mj-project\\mj-engine\\trained_models\\DQN_256_default_240120000035"
    # players.append(make_player("exp1"))
    game = Game(
        players=players,
        round_limit=args.round,
        game_limit=args.game,
        seed=args.seed,
        verbose=args.verbose)
    print(f"Game ready. Time elapsed: {time.process_time() - start:.3f}s")
    game.play()
    print(f"Game finished. Time elapsed: {time.process_time() - start:.3f}s")
    print(game.score_summary(detail=1))
