class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        """
        Removes the minimum number of parentheses to make the input string valid.
        Returns the resulting string.
        """
        stack = []
        to_remove = set()

        # First pass: mark indices of unmatched ')'
        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if stack:
                    stack.pop()
                else:
                    to_remove.add(i)

        # Add indices of unmatched '('
        to_remove.update(stack)

        # Build result string, skipping indices in to_remove
        return "".join([c for i, c in enumerate(s) if i not in to_remove])


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.minRemoveToMakeValid("lee(t(c)o)de)") == "lee(t(c)o)de"
    # Example 2
    assert sol.minRemoveToMakeValid("a)b(c)d") == "ab(c)d"
    # Example 3
    assert sol.minRemoveToMakeValid("))((") == ""
    # Example 4
    assert sol.minRemoveToMakeValid("(a(b(c)d)") == "a(b(c)d)"
    # Example 5: No parentheses
    assert sol.minRemoveToMakeValid("abcde") == "abcde"
    # Example 6: All valid
    assert sol.minRemoveToMakeValid("(a)(b)(c)") == "(a)(b)(c)"
    print("All tests passed.")


if __name__ == "__main__":
    test()
