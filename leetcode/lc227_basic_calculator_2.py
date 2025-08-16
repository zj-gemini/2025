from typing import List


class Solution:
    def calculate(self, s: str) -> int:
        """
        Evaluates a simple expression string containing non-negative integers, '+', '-', '*', '/', and spaces.
        Division truncates toward zero.
        """
        stack = []
        num = 0  # Current number being processed
        sign = "+"  # Previous operator, initialized to '+'
        s = s.replace(" ", "")  # Remove spaces for easier parsing
        n = len(s)
        for i, c in enumerate(s):
            if c.isdigit():
                num = num * 10 + int(c)  # Build the current number
            # If c is an operator or at the end of the string, process the previous number and sign
            if not c.isdigit() or i == n - 1:
                if sign == "+":
                    stack.append(num)  # Add the number to the stack
                elif sign == "-":
                    stack.append(-num)  # Add the negative number to the stack
                elif sign == "*":
                    stack.append(
                        stack.pop() * num
                    )  # Multiply top of stack with current number
                elif sign == "/":
                    prev = stack.pop()
                    # Python division truncates toward negative infinity, so use int() for truncation toward zero
                    stack.append(int(prev / num))
                sign = c  # Update the sign to the current operator
                num = 0  # Reset current number
        # The result is the sum of all numbers in the stack
        return sum(stack)


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.calculate("3+2*2") == 7
    # Example 2
    assert sol.calculate(" 3/2 ") == 1
    # Example 3
    assert sol.calculate(" 3+5 / 2 ") == 5
    # Example 4: Multiple operations
    assert sol.calculate("14-3/2") == 13
    # Example 5: Negative result
    assert sol.calculate("2-5*3") == -13
    # Example 6: Only one number
    assert sol.calculate("42") == 42
    # Example 7: Division truncates toward zero
    assert sol.calculate("7/3") == 2
    assert sol.calculate("-7/3") == -2
    print("All tests passed.")


if __name__ == "__main__":
    test()
