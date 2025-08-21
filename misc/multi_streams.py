from collections import deque
import heapq
from dataclasses import dataclass


@dataclass(order=True)
class Item:
    id: int
    val: float


class Stream:
    """A simple stream wrapper for a queue of Item objects."""

    def __init__(self, data: list[Item]):
        self.data = deque(data)

    def pop(self) -> Item | None:
        return self.data.popleft() if self.data else None


def read_streams(streams: list[Stream]) -> None:
    """
    Merge multiple sorted streams and print each (id, value) in order.
    """

    def maybe_add_available_stream(avail: list, idx: int) -> None:
        if (next_item := streams[idx].pop()) is not None:
            heapq.heappush(avail, (next_item, idx))

    avail_s = []
    for i in range(len(streams)):
        maybe_add_available_stream(avail_s, i)

    while avail_s:
        item, s_idx = heapq.heappop(avail_s)
        print(item.id, item.val)
        maybe_add_available_stream(avail_s, s_idx)


if __name__ == "__main__":
    ss = [
        Stream([Item(1, 2), Item(1, 0), Item(2, 1), Item(3, 1)]),
        Stream([Item(2, 2), Item(3, 2)]),
        Stream([Item(3, 0), Item(3, 4)]),
    ]
    read_streams(ss)
