# Examples for number-related common Python libraries and operations

# 1. Built-in sorting and min/max
nums = [5, 2, 9, 1, 5, 6]
print(sorted(nums))  # [1, 2, 5, 5, 6, 9]
print(sorted(nums, reverse=True))  # [9, 6, 5, 5, 2, 1]
print(min(nums))  # 1
print(max(nums))  # 9
print(sum(nums))  # 28

# 2. Built-in math functions
import math

print(math.sqrt(16))  # 4.0
print(math.ceil(2.3))  # 3
print(math.floor(2.7))  # 2
print(math.factorial(5))  # 120
print(math.gcd(12, 18))  # 6
print(math.isclose(0.1 + 0.2, 0.3))  # True (floating point comparison)
print(math.log(8, 2))  # 3.0 (log base 2)
print(math.log10(1000))  # 3.0 (log base 10)
print(math.exp(2))  # 7.38905609893065 (e^2)
print(math.pow(2, 5))  # 32.0 (2 to the power of 5)
print(math.pi)  # 3.141592653589793
print(math.e)  # 2.718281828459045

# 3. statistics module
import statistics

data = [1, 2, 2, 3, 4]
print(statistics.mean(data))  # 2.4
print(statistics.median(data))  # 2
print(statistics.mode(data))  # 2
print(statistics.stdev(data))  # Standard deviation

# 4. random module
import random

print(random.randint(1, 10))  # Random integer between 1 and 10
print(random.choice(nums))  # Random element from nums
print(random.sample(nums, 3))  # 3 unique random elements
random.shuffle(nums)
print(nums)  # nums is now shuffled

# 5. fractions for rational numbers
from fractions import Fraction

f = Fraction(3, 4) + Fraction(1, 6)
print(f)  # 11/12

# 6. decimal for precise floating point arithmetic
from decimal import Decimal, getcontext

getcontext().prec = 4
d = Decimal("0.1") + Decimal("0.2")
print(d)  # 0.3

# 7. heapq for heaps (priority queues)
import heapq

heap = [5, 2, 9, 1]
heapq.heapify(heap)
print(heapq.heappop(heap))  # 1 (smallest element)
heapq.heappush(heap, 3)
print(heapq.nsmallest(2, heap))  # [2, 3]
