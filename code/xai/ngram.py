from collections import Counter
from collections import defaultdict


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

    max_n = max(nlist)
    ngram_dict = {n: Counter() for n in nlist}
    for i in range(len(document)):
        for n in range(1, max_n + 1):
            if i + n > len(document):
                break
            # 如果 n 在 nlist 中，则统计该 n-gram 的频率
            if n in ngram_dict:
                ngram = document[i : i + n]
                ngram_dict[n][ngram] += 1

    results = {}
    for n, counter in ngram_dict.items():
        if counter:
            results[n] = counter.most_common(1)[0][0]
        else:
            results[n] = None
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
