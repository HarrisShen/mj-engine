from mjengine.analyzer import Analyzer
from mjengine.tiles import tiles_to_hand

if __name__ == '__main__':
    n = 100_000
    samples = []
    with open("samples/sampled_hands_100k_v2.txt", "r") as f:
        for _ in range(n):
            tiles = [int(t) for t in f.readline().split()]
            samples.append(tiles)

    print(f"Testing {n} samples")

    with open("samples/sampled_hands_100k_results_calsht_dw.txt", "r") as f:
        results_cpp = [tuple(int(num) for num in f.readline().split()) for _ in range(n)]  # Gold standard

    analyzer = Analyzer()
    analyzer.prepare("./index/")

    results_py = [0 for _ in range(n)]
    for i in range(n):
        results_py[i] = analyzer.calculate(tiles_to_hand(samples[i]), 4)
        if results_py[i] != results_cpp[i]:
            print("Calculation result inconsistent:")
            print(f"[{i}]: {sorted(samples[i])}")
            print(f"py - {results_py[i]}, cpp - {results_cpp[i]}")

    n_correct = sum(results_py[i] == results_cpp[i] for i in range(n))
    accuracy = n_correct / n
    print(f"Correct results: {n_correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    if accuracy != 1.0:
        print(f"WARNING: Calculation from Python version is incorrect!")
