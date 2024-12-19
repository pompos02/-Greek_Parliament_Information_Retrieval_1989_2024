import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load the TF-IDF matrix
with open("pkl_files/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

if not isinstance(tfidf_matrix, csr_matrix):
    raise TypeError("TF-IDF matrix is not in a sparse format. Convert it to CSR or CSC before proceeding.")

# Define maximum number of components to examine
max_k = 150  # Choose a reasonably large k for analysis

# Perform SVD with max_k components
U, Sigma, VT = svds(tfidf_matrix, k=max_k, which='LM')

# Reverse the order of singular values (svds gives smallest first)
Sigma = Sigma[::-1]

# Analyze variance explained
explained_variance_ratio = Sigma ** 2 / np.sum(Sigma ** 2)
cumulative_variance = np.cumsum(explained_variance_ratio)

# Plot cumulative variance
import matplotlib.pyplot as plt

plt.plot(range(1, len(Sigma) + 1), cumulative_variance, marker='o')
plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Explained')
plt.xlabel('Number of Themes')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by LSI Components')
plt.legend()
plt.show()
