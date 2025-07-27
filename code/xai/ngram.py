from collections import Counter


def find_most_common_char_ngrams(document: str, nlist: list[int]) -> dict[int, str]:
    """
    计算给定字符串中最常见的字符 n-gram。

    Args:
        document: 输入的原始字符串。
        nlist: 一个整数列表，指定要计算的 n-gram 的大小。

    Returns:
        一个字典，键是 n，值是该 n 值下最常见的 n-gram 字符串。
    """
    if not document or not nlist:
        return {}

    results = {}

    # 遍历 nlist 中的每一个 n
    for n in nlist:
        # 边界条件：如果字符串长度小于 n，无法形成 n-gram
        if n <= 0 or len(document) < n:
            results[n] = None
            continue

        # 使用列表推导式和切片生成所有 n-grams
        # 注意：这里直接操作字符串，无需分词
        ngrams = [document[i : i + n] for i in range(len(document) - n + 1)]

        if not ngrams:
            results[n] = None
            continue

        # 使用 collections.Counter 高效统计频率
        ngram_counts = Counter(ngrams)

        # 找到最常见的一个 n-gram 字符串
        most_common_item = ngram_counts.most_common(1)[0][0]
        results[n] = most_common_item

    return results


# --- 示例 ---
doc = "abracadabra"
n_values = [2, 3, 4]

most_common = find_most_common_char_ngrams(doc, n_values)
print(f"字符串: '{doc}'")
print(f"N值列表: {n_values}")
print("---" * 10)
print("计算结果:")
for n, ngram in most_common.items():
    print(f"最常见的 {n}-gram 是: '{ngram}'")

# 输出:
# 字符串: 'abracadabra'
# N值列表: [2, 3, 4]
# ------------------------------
# 计算结果:
# 最常见的 2-gram 是: 'ab'
# 最常见的 3-gram 是: 'abr'
# 最常见的 4-gram 是: 'abra'
