from sortedcontainers import SortedList
from collections import defaultdict


class Allocator:

    def __init__(self, n: int):
        self.freeBlocks = SortedList([(0, n - 1)])
        self.busyBlocks = defaultdict(list)

    def allocate(self, size: int, mID: int) -> int:
        for start, end in self.freeBlocks:
            if (blockEnd := start + size - 1) > end:
                continue  # block size is not enough
            self.busyBlocks[mID].append((start, blockEnd))
            self.freeBlocks.remove((start, end))
            if (
                blockEnd < end
            ):  # have free size in the block, return new free size to freeBlocks list
                self.freeBlocks.add((blockEnd + 1, end))
            return start

        return -1  # no avaialbe blocks

    def freeMemory(self, mID: int) -> int:
        freeBlocksCnt = 0
        for start, end in self.busyBlocks[mID]:
            freeBlocksCnt += end - start + 1
            idx = self.freeBlocks.bisect_left((start, end))

            # first process block from the right, because after removing from the left
            # all blockes moved to the -1 position in self.freeBlocks
            if idx < len(self.freeBlocks):  # check next free block if it does exist
                nextStart, nextEnd = self.freeBlocks[idx]
                if end + 1 == nextStart:
                    end = nextEnd
                    self.freeBlocks.remove((nextStart, nextEnd))

            # check left (previous) free block if it does exist
            if idx > 0:
                prevStart, prevEnd = self.freeBlocks[idx - 1]
                if prevEnd + 1 == start:
                    start = prevStart
                    self.freeBlocks.remove((prevStart, prevEnd))

            self.freeBlocks.add((start, end))
        del self.busyBlocks[mID]  # remove all released blocks

        return freeBlocksCnt


def test():
    loc = Allocator(10)
    print(loc.allocate(1, 1))  # 0
    print(loc.allocate(1, 2))  # 1
    print(loc.allocate(1, 3))  # 2
    print(loc.freeMemory(2))  # 1
    print(loc.allocate(3, 4))  # 3
    print(loc.allocate(1, 1))  # 1
    print(loc.allocate(1, 1))  # 6
    print(loc.freeMemory(1))  # 3
    print(loc.allocate(10, 2))  # -1
    print(loc.freeMemory(7))  # 0


# Uncomment to run tests
test()
