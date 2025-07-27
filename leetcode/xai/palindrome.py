def is_binary_palindrome(n: int) -> bool:
    b = bin(n)[2:]
    return b == b[::-1]


def generate_palindromes(N: int) -> list:
    palindromes = []
    # Add all single digit palindromes (1-9)
    for i in range(1, min(N + 1, 10)):
        if is_binary_palindrome(i):
            palindromes.append(i)
    # For numbers with more digits, generate palindromes by mirroring
    length = len(str(N))
    for l in range(2, length + 1):  # l is the length of the palindrome
        half_len = (l + 1) // 2  # Number of digits to generate for the left half
        start = 10 ** (half_len - 1)  # Smallest number for this half
        end = 10**half_len  # Largest number for this half (exclusive)
        for half in range(start, end):
            s = str(half)
            # For even length, mirror the whole half
            if l % 2 == 0:
                p = int(s + s[::-1])
            # For odd length, mirror all but the last digit of the half
            else:
                p = int(s + s[-2::-1])
            # Stop if palindrome exceeds N
            if p > N:
                break
            if is_binary_palindrome(p):
                palindromes.append(p)
    return palindromes


def test():
    print(
        generate_palindromes(100)
    )  # Only numbers <= 100 that are palindromes in both decimal and binary
    print(
        generate_palindromes(100000)
    )  # Only numbers <= 150 that are palindromes in both decimal and binary


# Uncomment to run tests
test()
