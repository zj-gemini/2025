class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        """
        Adds two non-negative integers represented as strings and returns the sum as a string.
        Does not convert the entire string to an integer directly.
        """
        n1, n2 = len(num1), len(num2)
        rst = []  # List to store result digits (in reverse order)
        carry = 0  # Carry for addition
        # Loop through each digit from right to left
        for i in range(max(n1, n2)):
            i1 = n1 - i - 1  # Index for num1
            i2 = n2 - i - 1  # Index for num2
            digit1 = int(num1[i1]) if i1 >= 0 else 0  # Get digit or 0 if out of bounds
            digit2 = int(num2[i2]) if i2 >= 0 else 0  # Get digit or 0 if out of bounds
            sum = digit1 + digit2 + carry  # Add digits and carry
            rst.append(str(sum % 10))  # Append the last digit of sum
            carry = sum // 10  # Update carry
        # If there's a carry left after the last addition, append it
        if carry:
            rst.append(str(carry))
        # The result is in reverse order, so reverse and join to form the final string
        return "".join(rst[::-1])


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.addStrings("11", "123") == "134"
    # Example 2
    assert sol.addStrings("456", "77") == "533"
    # Example 3
    assert sol.addStrings("0", "0") == "0"
    # Example 4: Different lengths
    assert sol.addStrings("1", "999") == "1000"
    # Example 5: Large numbers
    assert sol.addStrings("987654321", "123456789") == "1111111110"
    print("All tests passed.")


if __name__ == "__main__":
    test()
