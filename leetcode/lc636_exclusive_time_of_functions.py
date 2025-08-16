from typing import List


class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        """
        Returns the exclusive time of each function.
        Each log is a string: "{function_id}:{"start"|"end"}:{timestamp}"
        """
        result = [0] * n  # Stores exclusive time for each function
        stack = []  # Stack to keep track of function call order (stores function_id)
        prev_time = 0  # The previous timestamp processed

        for log in logs:
            # Parse the log entry
            fn_id_str, typ, time_str = log.split(":")
            fn_id = int(fn_id_str)
            timestamp = int(time_str)

            if typ == "start":
                if stack:
                    # The function on top of the stack was running until now
                    running_fn = stack[-1]
                    result[running_fn] += timestamp - prev_time
                # Start the new function
                stack.append(fn_id)
                prev_time = timestamp  # Update the previous time to the current start
            else:  # typ == "end"
                # The function on top of the stack ends now
                running_fn = stack.pop()
                # Add the time from prev_time to current timestamp (inclusive)
                result[running_fn] += timestamp - prev_time + 1
                prev_time = timestamp + 1  # Next function (if any) starts after this

        return result


# Unit tests
def test():
    sol = Solution()
    # Example 1
    n1 = 2
    logs1 = ["0:start:0", "1:start:2", "1:end:5", "0:end:6"]
    assert sol.exclusiveTime(n1, logs1) == [3, 4]
    # Example 2: Single function
    n2 = 1
    logs2 = ["0:start:0", "0:end:1"]
    assert sol.exclusiveTime(n2, logs2) == [2]
    # Example 3: Nested calls
    n3 = 3
    logs3 = ["0:start:0", "1:start:2", "1:end:5", "2:start:6", "2:end:9", "0:end:12"]
    assert sol.exclusiveTime(n3, logs3) == [7, 4, 4]
    # Example 4: Multiple calls to the same function
    n4 = 1
    logs4 = ["0:start:0", "0:end:0", "0:start:1", "0:end:2"]
    assert sol.exclusiveTime(n4, logs4) == [3]
    print("All tests passed.")


if __name__ == "__main__":
    test()
