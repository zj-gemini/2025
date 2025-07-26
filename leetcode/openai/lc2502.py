class Allocator:

    def __init__(self, n: int):
        # 0 means free
        self.mem = [0] * n

    def allocate(self, size: int, mID: int) -> int:
        if size <= 0:
            return -1
        n = len(self.mem)
        i = 0
        while i <= n - size:
            if all(self.mem[j] == 0 for j in range(i, i + size)):
                for j in range(i, i + size):
                    self.mem[j] = mID
                return i
            i += 1
        return -1

    def freeMemory(self, mID: int) -> int:
        count = 0
        for i in range(len(self.mem)):
            if self.mem[i] == mID:
                self.mem[i] = 0
                count += 1
        return count


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
#
