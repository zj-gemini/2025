# Examples for useful common containers in Python

# 2. collections.Counter
from collections import Counter

cnt = Counter(["a", "b", "a", "c", "b", "a"])
print(cnt)  # Counter({'a': 3, 'b': 2, 'c': 1})
print(cnt.most_common(2))  # [('a', 3), ('b', 2)]

# 3. collections.defaultdict
from collections import defaultdict

dd = defaultdict(list)
dd["a"].append(1)
dd["a"].append(2)
print(dd)  # defaultdict(<class 'list'>, {'a': [1, 2]})

# 4. collections.deque
from collections import deque

dq = deque([1, 2, 3])
dq.appendleft(0)
dq.append(4)
print(dq)  # deque([0, 1, 2, 3, 4])
dq.popleft()
print(dq)  # deque([1, 2, 3, 4])

# 5. collections.OrderedDict (Python 3.7+ dicts are ordered by default)
from collections import OrderedDict

od = OrderedDict()
od["a"] = 1
od["b"] = 2
print(list(od.keys()))  # ['a', 'b']

# 6. heapq (min-heap)
import heapq

heap = [3, 1, 4]
heapq.heapify(heap)
heapq.heappush(heap, 2)
print(heapq.heappop(heap))  # 1

# 7. sortedcontainers (third-party: pip install sortedcontainers)
from sortedcontainers import SortedList, SortedDict, SortedSet

# SortedList examples
sl = SortedList([3, 1, 2])
print(list(sl))  # [1, 2, 3]
sl.add(0)
print(list(sl))  # [0, 1, 2, 3]
sl.discard(2)
print(list(sl))  # [0, 1, 3]
print(sl.bisect_left(1))  # 1 (index where 1 would be inserted)
print(sl[-1])  # 3 (largest element)
print(sl[:2])  # [0, 1] (slicing works like a list)

# SortedDict examples
sd = SortedDict({"b": 2, "a": 1})
print(list(sd.items()))  # [('a', 1), ('b', 2)]
sd["c"] = 3
print(list(sd.keys()))  # ['a', 'b', 'c']
print(sd.peekitem(-1))  # ('c', 3) (last item)
del sd["b"]
print(list(sd.items()))  # [('a', 1), ('c', 3)]

# SortedSet examples
ss = SortedSet([3, 1, 2, 2])
print(list(ss))  # [1, 2, 3]
ss.add(0)
print(list(ss))  # [0, 1, 2, 3]
ss.discard(2)
print(list(ss))  # [0, 1, 3]
print(1 in ss)  # True
print(ss[-1])  # 3
