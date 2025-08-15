# Build adjacency list
from collections import defaultdict
import heapq


# Function to calculate the minimum cost and transaction path for transaction from b1 to b2. Note that, if b1 -> b3 -> b2 has lower transaction cost, we should use that.
def cost(
    b1: str, b2: str, transaction_cost: dict, amount: int
) -> tuple[float, list[str]]:
    adj = defaultdict(list)
    for (src, dst), (fee, threshold) in transaction_cost.items():
        adj[src].append((dst, fee, threshold))

    # Dijkstra's algorithm: (total_cost, path)
    heap = [(0, [b1])]
    visited = dict()  # bank -> min_cost

    while heap:
        cur_cost, path = heapq.heappop(heap)
        cur_bank = path[-1]
        if cur_bank in visited and visited[cur_bank] <= cur_cost:
            continue
        visited[cur_bank] = cur_cost
        if cur_bank == b2:
            return cur_cost, path
        for dest, fee, threshold in adj.get(cur_bank, []):
            # If amount > threshold, double the fee ratio
            fee_ratio = fee * (2 if amount > threshold else 1)
            next_cost = cur_cost + amount * fee_ratio
            heapq.heappush(heap, (next_cost, path + [dest]))
    return float("inf"), []  # No path found


# Here's the cost of transactions between banks (b1, b2, etc.). Also, the first number in the value tuple is the fee ratio, the second number is the threshold of doubling the fee ration.
transaction_cost = {
    ("b1", "b2"): (0.01, 150),
    ("b2", "b1"): (0.02, 200),
    ("b1", "b3"): (0.1, 150),
    ("b3", "b1"): (0.1, 150),
    ("b2", "b3"): (0.02, 150),
    ("b3", "b2"): (0.02, 250),
}


# Test cases for the cost function
def run_tests():
    tests = [
        ("b1", "b2", 100),
        ("b1", "b2", 200),
        ("b1", "b3", 210),
        ("b2", "b3", 100),
        ("b2", "b1", 250),
        ("b3", "b1", 100),
        ("b3", "b2", 300),
        ("b1", "b1", 50),  # same source and target
        ("b1", "b4", 100),  # non-existent target
    ]
    for src, dst, amt in tests:
        c, path = cost(src, dst, transaction_cost, amt)
        if c != float("inf"):
            c = round(c, 6)
        print(f"From {src} to {dst} with amount {amt}: cost={c}, path={path}")


if __name__ == "__main__":
    run_tests()
