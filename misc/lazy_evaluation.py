import sys

# --- Eager Evaluation (List Comprehension) ---
# All one million squares are calculated and stored in memory at once.
eager_list = [i * i for i in range(1_000_000)]
print(f"Eager List Size: {sys.getsizeof(eager_list)} bytes")

# --- Lazy Evaluation (Generator Expression) ---
# The generator object is created instantly. No squares are calculated yet.
lazy_generator = (i * i for i in range(1_000_000))
print(f"Lazy Generator Size: {sys.getsizeof(lazy_generator)} bytes")

# The values are computed only when we iterate over the generator.
# Let's get the first 5 values to show it works.
print("\nFirst 5 values from the lazy generator:")
for i in range(5):
    print(next(lazy_generator))


def fibonacci_generator(limit: int):
    """
    A generator function that yields Fibonacci numbers up to a limit.
    It doesn't calculate the whole sequence at once.
    """
    a, b = 0, 1
    while a < limit:
        print(f"(Yielding {a})")
        yield a  # Pause and return the current value of 'a'
        a, b = b, a + b  # Resume from here on the next call


print("\n--- Fibonacci Generator ---")
# Create the generator. The code inside fibonacci_generator has not run yet.
fib_gen = fibonacci_generator(100)
print("Generator created. Now let's iterate...")

# The function body only executes when we ask for a value.
for number in fib_gen:
    print(f"Received: {number}\n")


def read_large_file_lazy(file_path: str):
    """
    A generator to read a file line by line, yielding each line.
    This avoids loading the whole file into memory.
    """
    print(f"\n--- Reading '{file_path}' lazily ---")
    with open(file_path, "r") as f:
        for line in f:
            # yield the line, then pause until the next one is requested
            yield line.strip()


# Let's create a dummy large file for the example.
file_name = "large_log_file.txt"
with open(file_name, "w") as f:
    for i in range(5):
        f.write(f"Log entry number {i+1}\n")

# Create the generator to read the file.
log_reader = read_large_file_lazy(file_name)

# Process the file one line at a time.
# Only one line is held in memory at any given moment within the loop.
for log_entry in log_reader:
    print(f"Processing: '{log_entry}'")
