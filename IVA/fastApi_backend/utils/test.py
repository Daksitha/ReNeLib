import numpy as np

# Assuming you have these three arrays:
n = 4
A = np.random.rand(n, 2)
B = np.random.rand(n, 1)
C = np.random.rand(n, 1)
print("A",A)
print("B",B)
print("C",C)

# Concatenate B and C along the second axis (axis=1)
BC = np.concatenate((B, C), axis=1)

# Concatenate the result (BC) with A along the second axis (axis=1)
result = np.concatenate((A, BC), axis=1)

print(result)  # Output: (64, 58)
