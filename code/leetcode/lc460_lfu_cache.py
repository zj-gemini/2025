from collections import defaultdict, OrderedDict


class LFUCache:
    def __init__(self, capacity: int):
        # Maximum number of items the cache can hold
        self.capacity = capacity
        # Maps key to its value
        self.key_to_val = {}
        # Maps key to its frequency count
        self.key_to_freq = {}
        # Maps frequency to an OrderedDict of keys (for LRU among ties)
        self.freq_to_keys = defaultdict(OrderedDict)
        # Tracks the minimum frequency in the cache
        self.min_freq = 0

    def get(self, key: int) -> int:
        # Return -1 if key not present
        if key not in self.key_to_val:
            return -1
        # Update frequency since key is accessed
        self._update_freq(key)
        return self.key_to_val[key]

    def put(self, key: int, value: int) -> None:
        # Do nothing if capacity is zero
        if self.capacity == 0:
            return
        # If key exists, update value and frequency
        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._update_freq(key)
            return
        # If cache is full, evict LFU key (LRU among ties)
        if len(self.key_to_val) >= self.capacity:
            lfu_keys = self.freq_to_keys[self.min_freq]
            # Pop the least recently used key among those with min_freq
            evict_key, _ = lfu_keys.popitem(last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]
        # Insert new key with frequency 1
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1  # Reset min_freq to 1 for new key

    def _update_freq(self, key):
        # Helper to update frequency of a key
        freq = self.key_to_freq[key]
        # Remove key from current frequency's OrderedDict
        del self.freq_to_keys[freq][key]
        # If no keys left at this frequency, remove the dict and update min_freq
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        # Add key to next higher frequency's OrderedDict
        self.key_to_freq[key] = freq + 1
        self.freq_to_keys[freq + 1][key] = None


# Unit tests
import unittest


class TestLFUCache(unittest.TestCase):
    def test_example(self):
        # Example from LeetCode description
        lfu = LFUCache(2)
        lfu.put(1, 1)  # cache=[1,_], cnt(1)=1
        lfu.put(2, 2)  # cache=[2,1], cnt(2)=1, cnt(1)=1
        self.assertEqual(lfu.get(1), 1)  # cache=[1,2], cnt(1)=2
        lfu.put(3, 3)  # evicts key 2, cache=[3,1], cnt(3)=1, cnt(1)=2
        self.assertEqual(lfu.get(2), -1)
        self.assertEqual(lfu.get(3), 3)  # cache=[3,1], cnt(3)=2
        lfu.put(4, 4)  # evicts key 1, cache=[4,3], cnt(4)=1, cnt(3)=2
        self.assertEqual(lfu.get(1), -1)
        self.assertEqual(lfu.get(3), 3)  # cache=[3,4], cnt(3)=3
        self.assertEqual(lfu.get(4), 4)  # cache=[4,3], cnt(4)=2

    def test_zero_capacity(self):
        # Cache with zero capacity should never store anything
        lfu = LFUCache(0)
        lfu.put(1, 1)
        self.assertEqual(lfu.get(1), -1)

    def test_update(self):
        # Test updating value and eviction order
        lfu = LFUCache(2)
        lfu.put(1, 1)
        lfu.put(2, 2)
        lfu.put(1, 10)  # update value for key 1
        self.assertEqual(lfu.get(1), 10)
        lfu.put(3, 3)  # evicts key 2
        self.assertEqual(lfu.get(2), -1)


if __name__ == "__main__":
    unittest.main()
