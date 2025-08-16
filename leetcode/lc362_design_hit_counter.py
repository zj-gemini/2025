from collections import deque
from sortedcontainers import SortedList
from dataclasses import dataclass


@dataclass(order=True)
class Item:
    timestamp: int
    count: int = 1


class HitCounter:
    def __init__(self):
        # Store (timestamp, count) pairs in a queue
        self.q = SortedList([])
        self.total = 0

    def hit(self, timestamp: int) -> None:
        """
        Record a hit at the given timestamp.
        If the last hit was at the same timestamp, increment its count.
        Otherwise, append a new (timestamp, 1) entry.
        """
        if self.q and self.q[-1].timestamp == timestamp:
            self.q[-1].count += 1
        else:
            self.q.add(Item(timestamp))
        self.total += 1

    def getHits(self, timestamp: int) -> int:
        """
        Return the number of hits in the past 5 minutes (300 seconds).
        Remove outdated hits from the queue.
        """
        print("time:", timestamp)
        if self.q:
            min_idx_to_keep = self.q.bisect_left(Item(timestamp - 299))
            if min_idx_to_keep > 0:
                new_q = []
                self.total = 0
                for i in range(min_idx_to_keep, len(self.q)):
                    self.total += self.q[i].count
                    new_q.append(self.q[i])
                self.q = SortedList(new_q)
        return self.total


class HitCounterSimple:
    def __init__(self):
        # Store (timestamp, count) pairs in a queue
        self.q = deque()
        self.total = 0  # Total hits in the last 5 minutes

    def hit(self, timestamp: int) -> None:
        """
        Record a hit at the given timestamp.
        If the last hit was at the same timestamp, increment its count.
        Otherwise, append a new (timestamp, 1) entry.
        """
        if self.q and self.q[-1][0] == timestamp:
            # Increment count for the same timestamp
            self.q[-1][1] += 1
        else:
            self.q.append([timestamp, 1])
        self.total += 1

    def getHits(self, timestamp: int) -> int:
        """
        Return the number of hits in the past 5 minutes (300 seconds).
        Remove outdated hits from the queue.
        """
        # Remove hits older than 300 seconds
        while self.q and self.q[0][0] <= timestamp - 300:
            self.total -= self.q[0][1]
            self.q.popleft()
        return self.total


# Unit tests
def test():
    counter = HitCounter()
    counter.hit(1)
    counter.hit(2)
    counter.hit(3)
    assert counter.getHits(4) == 3
    counter.hit(300)
    assert counter.getHits(300) == 4
    assert counter.getHits(301) == 3

    # Multiple hits at the same timestamp
    counter2 = HitCounter()
    counter2.hit(1)
    counter2.hit(1)
    counter2.hit(1)
    assert counter2.getHits(1) == 3
    assert counter2.getHits(301) == 0

    # Hits at the edge of the window
    counter3 = HitCounter()
    counter3.hit(1)
    counter3.hit(100)
    counter3.hit(200)
    counter3.hit(300)
    assert counter3.getHits(300) == 4
    assert counter3.getHits(301) == 3
    assert counter3.getHits(400) == 2
    assert counter3.getHits(500) == 1
    assert counter3.getHits(601) == 0

    print("All tests passed.")


if __name__ == "__main__":
    test()
