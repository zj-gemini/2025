from typing import Union


class Solution:
    def myPow(self, x: float, n: int) -> float:
        """
        Computes x raised to the power n (x^n) using recursion and fast exponentiation.
        Handles negative exponents by inverting x and making n positive.
        """
        if n == 0:
            return 1  # Base case: any number to the power 0 is 1
        if n < 0:
            # For negative exponents, invert x and use positive n
            return self.myPow(1 / x, -n)
        if n % 2 == 1:
            # If n is odd, reduce the problem by one and multiply by x
            return x * self.myPow(x, n - 1)
        # If n is even, recursively compute half the exponent
        half = self.myPow(x, n // 2)
        return half * half  # Square the result for even exponents


# Unit tests
def test():
    sol = Solution()
    # Example 1: 2^10 = 1024
    assert abs(sol.myPow(2.0, 10) - 1024.0) < 1e-6
    # Example 2: 2^-2 = 0.25
    assert abs(sol.myPow(2.0, -2) - 0.25) < 1e-6
    # Example 3: 2^0 = 1
    assert abs(sol.myPow(2.0, 0) - 1.0) < 1e-6
    # Example 4: (-2)^3 = -8
    assert abs(sol.myPow(-2.0, 3) + 8.0) < 1e-6
    # Example 5: 0^5 = 0
    assert abs(sol.myPow(0.0, 5) - 0.0) < 1e-6
    # Example 6: 1^1000 = 1
    assert abs(sol.myPow(1.0, 1000) - 1.0) < 1e-6
    print("All tests passed.")


if __name__ == "__main__":
    test()
