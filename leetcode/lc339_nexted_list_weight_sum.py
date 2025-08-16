from typing import List, Union


# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation.
class NestedInteger:
    def __init__(self, value: Union[int, List["NestedInteger"]] = None):
        if value is None:
            self._integer = None
            self._list = []
        elif isinstance(value, int):
            self._integer = value
            self._list = None
        else:
            self._integer = None
            self._list = value

    def isInteger(self) -> bool:
        return self._integer is not None

    def getInteger(self) -> int:
        return self._integer

    def getList(self) -> List["NestedInteger"]:
        return self._list


class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def dfs(nestedList: List[NestedInteger], depth: int):
            sum = 0
            for node in nestedList:
                if node.isInteger():
                    sum += node.getInteger() * depth
                else:
                    sum += dfs(node.getList(), depth + 1)
            return sum

        sum = 0
        return dfs(nestedList, 1)


# Helper to build NestedInteger from nested Python lists
def build_nested_list(data):
    if isinstance(data, int):
        return NestedInteger(data)
    return NestedInteger([build_nested_list(x) for x in data])


# Unit tests
def test():
    sol = Solution()
    # Example 1: [[1,1],2,[1,1]] => 1*2 + 1*2 + 2*1 + 1*2 + 1*2 = 10
    nested1 = [
        build_nested_list([1, 1]),
        build_nested_list(2),
        build_nested_list([1, 1]),
    ]
    assert sol.depthSum(nested1) == 10
    # Example 2: [1,[4,[6]]] => 1*1 + 4*2 + 6*3 = 1 + 8 + 18 = 27
    nested2 = [build_nested_list(1), build_nested_list([4, [6]])]
    assert sol.depthSum(nested2) == 27
    # Example 3: [0] => 0*1 = 0
    nested3 = [build_nested_list(0)]
    assert sol.depthSum(nested3) == 0
    # Example 4: [] => 0
    nested4 = []
    assert sol.depthSum(nested4) == 0
    print("All tests passed.")


if __name__ == "__main__":
    test()
