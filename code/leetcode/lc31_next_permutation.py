from typing import List


class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Rearranges nums into the next lexicographical permutation.
        If not possible, rearranges to the lowest possible order (sorted in ascending order).

        1. 为什么要从右往左找“第一个升序对”？
        假设 nums 是一个排列。
        如果我们从右往左看，右边那一段一旦是降序，就意味着它已经是该位置所有可能的最大排列了。
        例如 ... 5, 4, 3, 1，这几个数字的最大字典序就是降序排列。
        想要得到比当前排列“刚好大一点”的排列，必须在左边某个地方做改变，让右边重新安排成最小的可能。
        这个“某个地方”就是第一个满足 nums[i] < nums[i+1] 的位置，因为：
        如果 nums[i] >= nums[i+1] 一直成立，那就说明后面这段是降序，没有办法在不动更左边的情况下变大。

        2. 为什么要找“右边第一个比 nums[i] 大的数字”？
        你在位置 i 想让整体变大，但又不能变得太大（要最接近的下一个）。
        在右侧降序区间里，第一个比 nums[i] 大的数就是刚好比它大一点的候选，这样交换后变化最小。
        如果你选了更大的数，就会跳过很多排列，不再是“next”。

        3. 为什么交换后要反转右边？
        右边在原来是降序的（因为我们是从右找的第一个升序点），交换后右边的排列依旧是降序。
        我们希望下一个排列是“变大最少”的，所以右边应该变成最小的升序排列。
        对降序序列反转，就得到了最小的升序。

        4. 一个直观例子（你会看到它和查字典一模一样）
        假设字典里有：
        ...  1 4 7 6 3 2
        从右往左找，4 < 7 在 i=1 处停下。
        右边是 [7, 6, 3, 2]，它已经是最大排列（降序）。
        在右边找一个刚好比 4 大的数，就是 6（不是 7，因为 6 比 4 大，但比 7 小，更接近）。
        交换后得到：
        ...  1 6 7 4 3 2
        最后，把右边 [7, 4, 3, 2] 反转成 [2, 3, 4, 7]：
        ...  1 6 2 3 4 7
        这就是在“字典”里 147632 的下一个词。

        5. 本质一句话
        next permutation 的算法本质就是：
        固定尽可能长的前缀（不变），在它的末尾做最小的递增调整，然后让后缀变成最小可能的排列。
        """
        n = len(nums)
        # 1. Find the first index 'i' from the end where nums[i] < nums[i+1]
        i = n - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1

        if i == -1:
            # The array is in descending order, just reverse it to get the smallest permutation
            nums.reverse()
            return

        # 2. Find the first index 'j' from the end where nums[j] > nums[i]
        j = n - 1
        while nums[j] <= nums[i]:
            j -= 1

        # 3. Swap nums[i] and nums[j]
        nums[i], nums[j] = nums[j], nums[i]

        # 4. Reverse the subarray nums[i+1:] to get the next smallest lexicographical order
        left, right = i + 1, n - 1
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1


# Unit tests
def test():
    sol = Solution()
    # Example 1
    nums1 = [1, 2, 3]
    sol.nextPermutation(nums1)
    assert nums1 == [1, 3, 2]
    # Example 2
    nums2 = [3, 2, 1]
    sol.nextPermutation(nums2)
    assert nums2 == [1, 2, 3]
    # Example 3
    nums3 = [1, 1, 5]
    sol.nextPermutation(nums3)
    assert nums3 == [1, 5, 1]
    # Example 4: Single element
    nums4 = [1]
    sol.nextPermutation(nums4)
    assert nums4 == [1]
    # Example 5: Two elements
    nums5 = [1, 2]
    sol.nextPermutation(nums5)
    assert nums5 == [2, 1]
    # Example 6: Already highest permutation
    nums6 = [2, 3, 1]
    sol.nextPermutation(nums6)
    assert nums6 == [3, 1, 2]
    print("All tests passed.")


if __name__ == "__main__":
    test()
