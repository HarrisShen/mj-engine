import argparse
import time

from mjengine.game import Game
from mjengine.player import make_player


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("players",
                        help="Player types or model directory/file if using a trained model", nargs=4)
    parser.add_argument("-r", "--round", help="Round limit of the game", type=int)
    parser.add_argument("-g", "--game", help="Limit of game to be played", type=int)
    parser.add_argument("-d", "--retain-dealer", default=False, action="store_true",
                        help="Enable dealer retaining in the game")
    parser.add_argument("-s", "--seed", help="Random seed used for game")
    parser.add_argument("-v", "--verbose", help="Verbosity of the program", action="count", default=0)
    args = parser.parse_args()

    print("Preparing game...")
    start = time.process_time()
    players = [make_player(args.players[i]) for i in range(4)]
    game = Game(
        players=players,
        round_limit=args.round,
        game_limit=args.game,
        retain_dealer=args.retain_dealer,
        seed=args.seed,
        verbose=args.verbose)
    print(f"Game ready. Time elapsed: {time.process_time() - start:.3f}s")
    game.play()
    print(f"Game finished. Time elapsed: {time.process_time() - start:.3f}s")
    print(game.score_summary(detail=1))
