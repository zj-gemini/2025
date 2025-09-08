# Leetcode 188: Best time to buy and sell stock IV

# You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.

# Find the maximum profit you can achieve. You may complete at most k transactions: i.e. you may buy at most k times and sell at most k times.

# Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

from typing import List


class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        """
        Finds the maximum profit with at most k transactions.

        This solution uses a recursive approach with memoization (top-down DP).
        The state is defined by (i, k, can_buy):
        - i: The current day (index in the prices array).
        - k: The number of transactions remaining.
        - can_buy: A boolean flag indicating if we are allowed to buy (True) or
                   must sell (False).

        The recurrence relation is:
        - If we can buy:
          - Buy: -prices[i] + solve(i + 1, k, False)
          - Do nothing: solve(i + 1, k, True)
        - If we must sell:
          - Sell: prices[i] + solve(i + 1, k - 1, True) (a transaction is completed)
          - Do nothing: solve(i + 1, k, False)

        The final answer is the maximum of these choices at each step.
        """
        n = len(prices)
        memo = {}  # Memoization table: (i, k, can_buy) -> max_profit

        def solve(i: int, k: int, can_buy: bool) -> int:
            # Base cases
            if i >= n or k == 0:
                return 0

            # Check memoization table
            if (i, k, can_buy) in memo:
                return memo[(i, k, can_buy)]

            # Recursive step
            if can_buy:
                # Option 1: Buy the stock at price[i]
                buy_profit = -prices[i] + solve(i + 1, k, False)
                # Option 2: Do nothing and move to the next day
                cooldown_profit = solve(i + 1, k, True)
                result = max(buy_profit, cooldown_profit)
            else:  # We must sell
                # Option 1: Sell the stock at price[i] (completes one transaction)
                sell_profit = prices[i] + solve(i + 1, k - 1, True)
                # Option 2: Do nothing and move to the next day
                cooldown_profit = solve(i + 1, k, False)
                result = max(sell_profit, cooldown_profit)

            # Store result in memoization table
            memo[(i, k, can_buy)] = result
            return result

        # Initial call: start at day 0 with k transactions, in a "can_buy" state.
        return solve(0, k, True)


# --- Unit Tests ---
def test_max_profit():
    sol = Solution()

    # Example 1: k=2, prices=[2,4,1] -> Profit: (4-2) = 2
    assert sol.maxProfit(2, [2, 4, 1]) == 2, "Test Case 1 Failed"

    # Example 2: k=2, prices=[3,2,6,5,0,3] -> Profit: (6-2) + (3-0) = 7
    assert sol.maxProfit(2, [3, 2, 6, 5, 0, 3]) == 7, "Test Case 2 Failed"

    # Edge case: No prices
    assert sol.maxProfit(2, []) == 0, "Test Case 3 Failed"

    # Edge case: k=0
    assert sol.maxProfit(0, [1, 2, 3]) == 0, "Test Case 4 Failed"

    # Special case: k is large (unlimited transactions)
    assert sol.maxProfit(10, [1, 2, 3, 4, 5]) == 4, "Test Case 5 Failed"

    # Complex case
    assert sol.maxProfit(2, [1, 2, 4, 2, 5, 7, 2, 4, 9, 0]) == 13, "Test Case 6 Failed"
    # Transactions: (4-1) + (9-2) = 3 + 7 = 10? No.
    # (2-1) + (7-2) = 1 + 5 = 6? No.
    # (7-1) + (9-2) = 6 + 7 = 13. Yes.

    print("All test cases passed!")


if __name__ == "__main__":
    test_max_profit()
