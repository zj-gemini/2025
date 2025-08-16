# Example code for popular Python string manipulation libraries

# 1. Built-in string methods
s = "  Hello, World!  "
print(s.strip())  # Remove leading/trailing whitespace
print(s.lower())  # Convert to lowercase
print(s.replace("World", "Python"))  # Replace substring
print(s.split(","))  # Split by comma
print(s.find("World"))  # Find substring position (returns 8)
print(s.find("Python"))  # Substring not found (returns -1)
print(s.find("l"))  # First occurrence of 'l' (returns 3)
print(s.find("l", 5))  # Find 'l' starting from index 5 (returns 10)
print(s.find("o", 0, 8))  # Find 'o' between index 0 and 8
print("-".join(["a", "b", "c"]))  # Join list into string

# Check if string is alphabetic, numeric, alphanumeric, etc.
print("Hello".isalpha())  # True (all letters)
print("123".isdigit())  # True (all digits)
print("Hello123".isalnum())  # True (letters and digits)
print("   ".isspace())  # True (all whitespace)
print("hello".islower())  # True
print("HELLO".isupper())  # True
print("Hello World".istitle())  # True
print("hello".startswith("he"))  # True
print("hello".endswith("lo"))  # True

# 2. re (regular expressions)
import re

text = "The rain in Spain"
print(re.findall(r"\b\w{4}\b", text))  # Find all 4-letter words
print(re.sub(r"ain", "___", text))  # Replace 'ain' with '___'
print(re.match(r"The", text))  # Match at the beginning
print(re.split(r"\s", text))  # Split by whitespace
print(re.search(r"rain", text))  # Search for 'rain' anywhere

text = "apple,banana;orange grape"
# Split on comma, semicolon, or space
print(re.split(r"[;, ]+", text))  # Output: ['apple', 'banana', 'orange', 'grape']

# 3. string (standard library)
import string

print(string.ascii_lowercase)  # 'abcdefghijklmnopqrstuvwxyz'
print(string.digits)  # '0123456789'
print(string.punctuation)  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
print(string.capwords("hello world! this is python."))  # Capitalize words

# 4. textwrap (for wrapping and filling text)
import textwrap

long_text = "This is a very long sentence that needs to be wrapped."
print(textwrap.fill(long_text, width=20))
print(textwrap.shorten("Python is awesome and powerful!", width=15, placeholder="..."))
print(textwrap.indent("Line1\nLine2", prefix=">> "))  # Add prefix to each line

# 5. difflib (for comparing strings)
import difflib

a = "apple"
b = "apricot"
print(list(difflib.ndiff(a, b)))  # Show differences between two strings
print(difflib.SequenceMatcher(None, a, b).ratio())  # Similarity ratio
print(
    list(difflib.unified_diff(["foo\n", "bar\n"], ["foo\n", "baz\n"]))
)  # Unified diff

# 6. unicodedata (for Unicode normalization)
import unicodedata

s = "café"
print(unicodedata.normalize("NFD", s))  # Decompose accented characters
print([unicodedata.name(c) for c in s])  # Unicode names for each character
print(unicodedata.category("é"))  # Unicode category
