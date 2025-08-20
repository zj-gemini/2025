import numpy as np

# 1. Create arrays
a = np.array([1, 2, 3])
b = np.zeros((2, 3))  # 2x3 array of zeros
c = np.ones((3, 2))  # 3x2 array of ones
d = np.eye(3)  # 3x3 identity matrix
e = np.arange(0, 10, 2)  # [0 2 4 6 8]
f = np.linspace(0, 1, 5)  # 5 numbers from 0 to 1

# 2. Array shape and type
print(a.shape)  # (3,)
print(b.dtype)  # float64

# 3. Reshape and flatten
g = np.arange(6).reshape(2, 3)  # 2x3 array
print(g.flatten())  # [0 1 2 3 4 5]

# 4. Indexing and slicing
print(a[1])  # 2
print(g[1, 2])  # 5
print(g[:, 1])  # [1 4]

# 5. Basic operations
print(a + 10)  # [11 12 13]
print(a * 2)  # [2 4 6]
print(a + np.array([4, 5, 6]))  # [5 7 9]
print(np.sum(g))  # 15
print(np.mean(g))  # 2.5

# 6. Matrix multiplication
h = np.array([[1, 2], [3, 4]])
i = np.array([[5, 6], [7, 8]])
print("Dot product:")
print(np.dot(h, i))
print("\nMatmul operator @:")
print(h @ i)  # Same as np.matmul for 2D arrays
print("\nnp.matmul function:")
print(np.matmul(h, i))  # Equivalent to @

# For higher dimensional arrays, matmul performs broadcasting and stacks matrix multiplications
# Create two stacks of 2x3 and 3x2 matrices
j = np.arange(12).reshape(2, 2, 3)
k = np.arange(12).reshape(2, 3, 2)
print("\nMatmul on 3D arrays (stack of matrices):")
print(np.matmul(j, k))
# 7. Transpose and axis operations
print(h.T)  # Transpose
print(np.sum(h, axis=0))  # Sum over columns
print(np.sum(h, axis=1))  # Sum over rows

# 8. Boolean indexing and filtering
arr = np.array([1, 2, 3, 4, 5])
print(arr[arr > 2])  # [3 4 5]

# 9. Random numbers
np.random.seed(0)
print(np.random.rand(2, 3))  # 2x3 array of random floats in [0, 1)
print(np.random.randint(0, 10, (2, 2)))  # 2x2 array of random ints

# 10. Save and load arrays
np.save("my_array.npy", a)
a_loaded = np.load("my_array.npy")
print(a_loaded)
