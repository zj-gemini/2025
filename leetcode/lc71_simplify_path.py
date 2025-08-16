from typing import List


class Solution:
    def simplifyPath(self, path: str) -> str:
        """
        Simplifies a given Unix-style absolute file path.
        """
        stack = []
        parts = path.split("/")
        for part in parts:
            if part == "" or part == ".":
                continue  # Ignore empty and current directory
            elif part == "..":
                if stack:
                    stack.pop()  # Go up one directory
            else:
                stack.append(part)  # Valid directory name
        return "/" + "/".join(stack)


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.simplifyPath("/home/") == "/home"
    # Example 2
    assert sol.simplifyPath("/../") == "/"
    # Example 3
    assert sol.simplifyPath("/home//foo/") == "/home/foo"
    # Example 4
    assert sol.simplifyPath("/a/./b/../../c/") == "/c"
    # Example 5
    assert sol.simplifyPath("/a/../../b/../c//.//") == "/c"
    # Example 6
    assert sol.simplifyPath("/a//b////c/d//././/..") == "/a/b/c"
    # Example 7: Only root
    assert sol.simplifyPath("/") == "/"
    print("All tests passed.")


if __name__ == "__main__":
    test()
