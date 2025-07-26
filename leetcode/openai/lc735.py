from typing import List


class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        stack = []
        for a in asteroids:
            while stack and a < 0 < stack[-1]:
                if stack[-1] < -a:
                    stack.pop()
                    continue
                elif stack[-1] == -a:
                    stack.pop()
                break
            else:
                stack.append(a)
        return stack


def test():
    sol = Solution()
    print(sol.asteroidCollision([-2, 1, -2, -2]))  # [-2,-2,-2]
    print(sol.asteroidCollision([10, 2, -5]))  # [10]
    print(sol.asteroidCollision([8, -8]))  # [5,10]
    print(sol.asteroidCollision([5, 10, -5]))  # [5,10]
    print(sol.asteroidCollision([8, -8]))  # []
    print(sol.asteroidCollision([-2, -1, 1, 2]))  # [-2,-1,1,2]
    print(sol.asteroidCollision([1, -1, -2, -2]))  # [-2,-2]
    print(sol.asteroidCollision([1, 2, 3, -3, -2, -1]))  # []


test()
