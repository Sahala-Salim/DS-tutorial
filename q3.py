import numpy as np

# Get matrix dimensions from user
rows_A = int(input("Enter number of rows for Matrix X: "))
cols_A = int(input("Enter number of columns for Matrix X: "))
rows_B = int(input("Enter number of rows for Matrix Y: "))
cols_B = int(input("Enter number of columns for Matrix Y: "))

# Check if matrix multiplication is possible
if cols_A != rows_B:
    print("Matrix multiplication not possible!")
    exit()

# Initialize matrices
X = []
Y = []

# Get user input for Matrix X
print("Enter values for Matrix X:")
for i in range(rows_A):
    row = [int(input()) for _ in range(cols_A)]
    X.append(row)

# Get user input for Matrix Y
print("Enter values for Matrix Y:")
for i in range(rows_B):
    row = [int(input()) for _ in range(cols_B)]
    Y.append(row)

# Initialize result matrix
Z = [[0] * cols_B for _ in range(rows_A)]

# Perform manual matrix multiplication
for i in range(rows_A):
    for j in range(cols_B):
        for k in range(cols_A):
            Z[i][j] += X[i][k] * Y[k][j]

# Display results
print("\nManual Multiplication Result:")
for row in Z:
    print(row)

print("\nNumPy Multiplication Result:")
print(np.dot(X, Y))
