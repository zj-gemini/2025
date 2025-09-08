from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n <= 1:
            return 0
        # first_half[i] represents that, with one trade, between [0, i), the max profit
        first_half = [0] * n
        min_price = prices[0]
        max_profit = 0
        for i in range(n):
            profit = prices[i] - min_price
            max_profit = max(profit, max_profit)
            first_half[i] = max_profit
            min_price = min(min_price, prices[i])
        # second_half[i] represents that, with one trade, between [i, n), the max profit
        second_half = [0] * n
        max_profit = 0
        max_price = prices[-1]
        for i in range(n):
            cur_i = n - i - 1
            profit = max_price - prices[cur_i]
            max_profit = max(max_profit, profit)
            second_half[cur_i] = max_profit
            max_price = max(max_price, prices[cur_i])
        max_profit = 0
        for i in range(n):
            max_profit = max(max_profit, first_half[i] + second_half[i])
        return max_profit


# --- Unit Tests ---
def test_max_profit():
    sol = Solution()

    assert sol.maxProfit([3, 3, 5, 0, 0, 3, 1, 4]) == 6, "Test Case 1 Failed"
    assert sol.maxProfit([1, 2, 3, 4, 5]) == 4, "Test Case 2 Failed"
    assert sol.maxProfit([7, 6, 4, 3, 1]) == 0, "Test Case 3 Failed"
    assert sol.maxProfit([1]) == 0, "Test Case 4 Failed"
    assert sol.maxProfit([2, 1, 4, 5, 2, 9, 7]) == 11, "Test Case 5 Failed"

    print("All test cases passed!")


if __name__ == "__main__":
    test_max_profit()
