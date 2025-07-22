from collections import defaultdict, deque, Counter


def alienOrder(words):
    # Build graph
    adj = defaultdict(set)
    indegree = Counter({c: 0 for word in words for c in word})

    for w1, w2 in zip(words, words[1:]):
        for c1, c2 in zip(w1, w2):
            if c1 != c2:
                if c2 not in adj[c1]:
                    adj[c1].add(c2)
                    indegree[c2] += 1
                break
        else:  # one word is the prefix of the other
            if len(w2) < len(w1):
                return ""

    # Topological sort
    queue = deque([c for c in indegree if indegree[c] == 0])
    res = []
    while queue:
        c = queue.popleft()
        res.append(c)
        for nei in adj[c]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)
    if len(res) != len(indegree):
        return ""
    return "".join(res)


def test():
    print(alienOrder(["wrt", "wrf", "er", "ett", "rftt"]))  # "wertf"
    print(alienOrder(["z", "x"]))  # "zx"
    print(alienOrder(["z", "x", "z"]))  # ""
    print(alienOrder(["abc", "ab"]))  # "" (invalid)
    print(alienOrder(["a", "b", "c"]))  # "abc"


# Uncomment to run tests
test()
