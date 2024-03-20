import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from tqdm import tqdm

from mjengine.game import Game
from mjengine.player import make_player, make_players


def run_games(opponent, model, seed):
    players = make_players([opponent] * 3 + [model])
    game = Game(
        players=players,
        round_limit=1024,
        game_limit=None,
        retain_dealer=False,
        seed=seed,
        verbose=0)
    with tqdm(total=1024, desc=f"Against {opponent}") as pbar:
        while not game.is_finished():
            game.play(rounds=1)
            pbar.update(1)
    return opponent, game


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="Model directory or state file")
    parser.add_argument("-l", "--level",
                        help="Highest level of tested components, order - r, r1, a, v, e0, e1",
                        type=int, choices=list(range(1, 7)), default=6)
    parser.add_argument("-s", "--seed", help="Random seed used for game")
    args = parser.parse_args()

    opponents = ["random", "random1", "analyzer", "value", "exp0", "exp1"]
    opponents = opponents[:args.level]
    results = {}
    with ThreadPoolExecutor(max_workers=len(opponents)) as ex:
        futures = [ex.submit(run_games, opponent, args.model, args.seed) for opponent in opponents]
        for future in as_completed(futures):
            oppo, game = future.result()
            results[oppo] = game

    for opponent in opponents:
        print(f"Against {opponent}")
        print(results[opponent].score_summary(detail=1))
