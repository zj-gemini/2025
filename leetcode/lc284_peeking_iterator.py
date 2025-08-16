from typing import Iterator, Any


class PeekingIterator:
    """
    An iterator that supports peeking at the next element without advancing.
    Compatible with LeetCode-style iterators that have next() and hasNext().
    """

    def __init__(self, iterator):
        # Store the underlying iterator and prefetch the first element
        self._next = iterator.next()
        self._iterator = iterator

    def peek(self):
        """Return the next element without advancing the iterator."""
        return self._next

    def next(self):
        """Return the next element and advance the iterator."""
        if self._next is None:
            raise StopIteration()
        to_return = self._next
        self._next = None
        if self._iterator.hasNext():
            self._next = self._iterator.next()
        return to_return

    def hasNext(self):
        """Return True if there are more elements to iterate."""
        return self._next is not None


# Custom iterator for LeetCode-style interface
class ListIterator:
    """
    A simple wrapper to mimic LeetCode's iterator interface for testing.
    Provides next() and hasNext() methods.
    """

    def __init__(self, nums):
        self.nums = nums
        self.index = 0  # Current position in the list

    def next(self):
        """Return the next element and advance the iterator."""
        if self.hasNext():
            val = self.nums[self.index]
            self.index += 1
            return val
        raise StopIteration()

    def hasNext(self):
        """Return True if there are more elements to iterate."""
        return self.index < len(self.nums)


def test() -> None:
    """
    Test PeekingIterator with a custom ListIterator.
    Demonstrates normal usage and edge cases.
    """
    nums = [1, 2, 3]
    it = PeekingIterator(ListIterator(nums))
    print(it.next())  # 1
    print(it.peek())  # 2
    print(it.next())  # 2
    print(it.next())  # 3
    print(it.hasNext())  # False

    # Edge case: peek/next after end
    try:
        print(it.peek())
    except StopIteration as e:
        print("Caught StopIteration on peek:", e)

    try:
        print(it.next())
    except StopIteration as e:
        print("Caught StopIteration on next:", e)


test()
