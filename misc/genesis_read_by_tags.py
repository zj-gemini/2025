# We store data


# O(N) lookup for read
class DataStoreSimple:
    def __init__(self):
        self._data: list[tuple[str, set[str]]] = []

    # Add a data point. The content is arbitrary string, with corresponding tags
    def add(self, content: str, tags: set[str]):
        self._data.append((content, tags))

    # Read all content with tags that are a super set of the argument tags.
    def read(self, tags: set[str]) -> list[str]:
        result = []
        for content, item_tags in self._data:
            if item_tags.issuperset(tags):
                result.append(content)
        return result


from collections import defaultdict


class DataStore:
    def __init__(self):
        self._data: list[tuple[str, set[str]]] = []
        self._tag_index: defaultdict[str, set[int]] = defaultdict(set)

    def add(self, content: str, tags: set[str]):
        idx = len(self._data)
        self._data.append((content, tags))
        for tag in tags:
            self._tag_index[tag].add(idx)

    def read(self, tags: set[str]) -> list[str]:
        """
        Return all content whose tags are a superset of the given tags.
        Uses an inverted index for efficient lookup: for each tag, retrieves the set of indices
        where that tag appears, then intersects these sets to find content that contains all tags.
        If tags is empty, returns all content.
        """
        if not tags:
            return [content for content, _ in self._data]
        # For each tag, get the set of indices where that tag appears
        tag_index_sets = [self._tag_index.get(tag, set()) for tag in tags]
        # If any tag is not present, or no tags given, return empty
        if not tag_index_sets or any(not s for s in tag_index_sets):
            return []
        # Intersect all sets to find indices present in all tag sets
        idxs = set.intersection(*tag_index_sets)
        # Return the content at those indices
        return [self._data[i][0] for i in idxs]


ds = DataStore()
ds.add("1", set(["t1"]))
ds.add("2", set(["t1", "t2"]))
ds.add("3", set(["t1", "t3"]))


assert set(ds.read({"t1"})) == {"1", "2", "3"}
assert set(ds.read({"t1", "t3"})) == {"3"}
assert set(ds.read({"t2", "t3"})) == set()
assert set(ds.read({})) == {"1", "2", "3"}
