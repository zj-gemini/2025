from typing import List

class Codec:
    def encode(self, strs: List[str]) -> str:
        # Use length + '#' + string for each element
        return ''.join(f"{len(s)}#{s}" for s in strs)

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0
        while i < len(s):
            j = s.find('#', i)
            length = int(s[i:j])
            res.append(s[j+1:j+1+length])
            i = j + 1 + length
        return res

def test():
    codec = Codec()
    cases = [
        (["Hello", "World"], ["Hello", "World"]),
        ([""], [""]),
        (["a", "b", "c"], ["a", "b", "c"]),
        (["#", "12#34", ""], ["#", "12#34", ""]),
        (["abc#def", "ghi"], ["abc#def", "ghi"]),
    ]
    for strs, expected in cases:
        encoded = codec.encode(strs)
        decoded = codec.decode(encoded)
        print(f"Input: {strs} | Encoded: {encoded} | Decoded: {decoded} | Pass: {decoded == expected}")

# Uncomment to run tests