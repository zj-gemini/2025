from collections import defaultdict, deque


def alienOrder(words):
    # Build graph
    adj = defaultdict(set)
    indegree = {}
    for word in words:
        for c in word:
            indegree[c] = 0

    for i in range(len(words) - 1):
        w1, w2 = words[i], words[i + 1]
        min_len = min(len(w1), len(w2))
        if w1[:min_len] == w2[:min_len] and len(w1) > len(w2):
            return ""
        for j in range(min_len):
            if w1[j] != w2[j]:
                if w2[j] not in adj[w1[j]]:
                    adj[w1[j]].add(w2[j])
                    indegree[w2[j]] += 1
                break

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
