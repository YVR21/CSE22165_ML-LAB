import pandas as pd
import numpy as np

# Load data
data = {
    'Customer': ['C_1', 'C_2', 'C_3', 'C_4', 'C_5', 'C_6', 'C_7', 'C_8', 'C_9', 'C_10'],
    'Candies (#)': [20, 16, 27, 19, 24, 22, 15, 18, 21, 16],
    'Mangoes (Kg)': [6, 3, 6, 1, 4, 1, 4, 4, 1, 2],
    'Milk Packets (#)': [2, 6, 2, 2, 2, 5, 2, 2, 4, 4],
    'Payment (Rs)': [386, 289, 393, 110, 280, 167, 271, 274, 148, 198]
}

# Create DataFrame
df = pd.DataFrame(data)

# Define Matrix A (features) and Matrix C (target)
A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df['Payment (Rs)'].values

# Dimensionality of the vector space
dimensionality = A.shape[1]

# Number of vectors in the vector space (rows in A)
num_vectors = A.shape[0]

# Rank of Matrix A
rank_A = np.linalg.matrix_rank(A)

# Compute the pseudo-inverse of Matrix A
A_pseudo_inv = np.linalg.pinv(A)

# Compute the cost of each product (weights) using pseudo-inverse
costs = A_pseudo_inv @ C

# Output results
print("Matrix A:")
print(A)
print("\nMatrix C:")
print(C)
print("\nDimensionality of the vector space:")
print(dimensionality)
print("\nNumber of vectors in the vector space:")
print(num_vectors)
print("\nRank of Matrix A:")
print(rank_A)
print("\nCost of each product (using pseudo-inverse):")
print(costs)
