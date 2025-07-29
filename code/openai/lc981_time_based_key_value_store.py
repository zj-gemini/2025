from collections import defaultdict
from sortedcontainers import SortedList


class TimeMap:
    """
    Time-based key-value store supporting set and get with timestamps.
    Uses binary search for efficient retrieval.
    """

    def __init__(self):
        # Store for each key: a sorted list of (timestamp, value) pairs, sorted by timestamp
        self.store = defaultdict(SortedList)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # Append the (timestamp, value) pair for the key
        self.store[key].add((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        # Retrieve the value with the largest timestamp <= given timestamp
        arr = self.store.get(key, [])
        if not arr:
            return ""
        # Use bisect_right to find the insertion point for timestamp
        i = arr.bisect_right((timestamp, chr(127)))
        if i == 0:
            return ""
        return arr[i - 1][1]


# Test function for the example case
def test():
    timeMap = TimeMap()
    timeMap.set("foo", "bar", 1)
    print(timeMap.get("foo", 1))  # "bar"
    print(timeMap.get("foo", 3))  # "bar"
    timeMap.set("foo", "bar2", 4)
    print(timeMap.get("foo", 4))  # "bar2"
    print(timeMap.get("foo", 5))  # "bar2"


if __name__ == "__main__":
    test()
