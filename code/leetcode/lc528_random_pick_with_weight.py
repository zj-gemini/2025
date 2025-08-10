import random
import bisect
from typing import List


class Solution:
    def __init__(self, w: List[int]):
        # Build prefix sum array for weights
        self.prefix = []
        total = 0
        for weight in w:
            total += weight
            self.prefix.append(total)
        self.total = total

    def pickIndex(self) -> int:
        # Pick a random integer in [1, total]
        target = random.randint(1, self.total)
        # Find the leftmost index such that prefix[index] >= target
        return bisect.bisect_left(self.prefix, target)


# Unit tests
def test():
    # Test with single element
    sol1 = Solution([1])
    results1 = [sol1.pickIndex() for _ in range(10)]
    assert all(r == 0 for r in results1)

    # Test with two elements, [1, 3]
    sol2 = Solution([1, 3])
    counts = [0, 0]
    for _ in range(10000):
        idx = sol2.pickIndex()
        counts[idx] += 1
    # Index 1 should be picked about 3 times as often as index 0
    ratio = counts[1] / counts[0]
    assert 2.5 < ratio < 3.5  # Allow some randomness

    # Test with more elements
    sol3 = Solution([2, 5, 3])
    counts = [0, 0, 0]
    for _ in range(10000):
        idx = sol3.pickIndex()
        counts[idx] += 1
    total = sum(counts)
    probs = [c / total for c in counts]
    # Probabilities should be roughly proportional to weights
    assert abs(probs[0] - 0.2) < 0.05
    assert abs(probs[1] - 0.5) < 0.05
    assert abs(probs[2] - 0.3) < 0.05

    print("All tests passed.")


if __name__ == "__main__":
    test()
