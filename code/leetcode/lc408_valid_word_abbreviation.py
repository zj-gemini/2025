class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        """
        Checks if abbr is a valid abbreviation for word.
        """
        i = 0  # Pointer for word
        j = 0  # Pointer for abbr

        while i < len(word) and j < len(abbr):
            if abbr[j].isalpha():
                # Match character
                if word[i] != abbr[j]:
                    return False
                i += 1
                j += 1
            else:
                # abbr[j] is a digit, parse the full number
                if abbr[j] == "0":
                    return False  # Leading zeros not allowed
                num = 0
                while j < len(abbr) and abbr[j].isdigit():
                    num = num * 10 + int(abbr[j])
                    j += 1
                i += num  # Skip num characters in word

        # Both pointers should reach the end
        return i == len(word) and j == len(abbr)


# Unit tests
def test():
    sol = Solution()
    # Example 1
    assert sol.validWordAbbreviation("internationalization", "i12iz4n") == True
    # Example 2
    assert sol.validWordAbbreviation("apple", "a2e") == False
    # No abbreviation
    assert sol.validWordAbbreviation("apple", "apple") == True
    # Full abbreviation
    assert sol.validWordAbbreviation("substitution", "12") == True
    # Leading zero
    assert sol.validWordAbbreviation("apple", "a01e") == False
    # Adjacent numbers not possible in abbr, but test anyway
    assert sol.validWordAbbreviation("substitution", "s10n") == True
    # Skipping too many
    assert sol.validWordAbbreviation("apple", "a5") == False
    # Skipping not enough
    assert sol.validWordAbbreviation("apple", "a3") == False
    # Abbreviation ends with number
    assert sol.validWordAbbreviation("apple", "a4") == True
    print("All tests passed.")


if __name__ == "__main__":
    test()
