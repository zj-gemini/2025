from typing import List

DELIMITER = "**"
LEN_DELIM = len(DELIMITER)


class Codec:
    def encode(self, strs: List[str]) -> str:
        """Encodes a list of strings to a single string."""
        # Encode as "length**string" for each string
        return "".join(f"{len(s)}{DELIMITER}{s}" for s in strs)

    def decode(self, s: str) -> List[str]:
        """Decodes a single string to a list of strings."""
        res = []
        i = 0
        while i < len(s):
            j = s.find(DELIMITER, i)
            if j < 0:
                raise ValueError("Invalid encoded string")
            length = int(s[i:j])
            i = j + LEN_DELIM + length
            res.append(s[j + LEN_DELIM : i])
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
        print(
            f"Input: {strs} | Encoded: {encoded} | Decoded: {decoded} | Pass: {decoded == expected}"
        )


# Uncomment to run tests
