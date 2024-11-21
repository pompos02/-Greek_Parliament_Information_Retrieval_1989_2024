import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os

# Load the TF-IDF matrix
with open("pkl_files/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

if not isinstance(tfidf_matrix, csr_matrix):
    raise TypeError("TF-IDF matrix is not in a sparse format. Convert it to CSR or CSC before proceeding.")

# Define range of themes to test
theme_range = [60, 80, 90, 100, 120]

# Initialize results
silhouette_scores = []
davies_bouldin_scores = []
inertia_values = []

# Variance explained analysis
explained_variances = {}

for n_components in theme_range:
    # Perform LSI
    U, Sigma, VT = svds(tfidf_matrix, k=n_components)
    U = U[:, ::-1]
    Sigma = Sigma[::-1]
    VT = VT[::-1, :]

    # Compute variance explained
    explained_variance_ratio = Sigma**2 / np.sum(Sigma**2)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    explained_variances[n_components] = cumulative_variance

    # Compute LSI matrix
    lsi_matrix = np.dot(U, np.diag(Sigma))
    lsi_matrix_normalized = normalize(lsi_matrix, norm='l2')

    # Evaluate clustering quality for a fixed number of clusters (e.g., k=10)
    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(lsi_matrix_normalized)

    silhouette_avg = silhouette_score(lsi_matrix_normalized, labels)
    db_score = davies_bouldin_score(lsi_matrix_normalized, labels)

    silhouette_scores.append((n_components, silhouette_avg))
    davies_bouldin_scores.append((n_components, db_score))

    print(f"n_components={n_components}: Silhouette Score={silhouette_avg:.3f}, Davies-Bouldin Index={db_score:.3f}")

    # Evaluate inertia for multiple cluster sizes
    for k in range(2, 20):  # Testing clusters from 2 to 20
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(lsi_matrix_normalized)
        inertia_values.append((n_components, k, kmeans.inertia_))

# Save results for reference
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
with open(os.path.join(output_folder, "clustering_metrics.pkl"), "wb") as f:
    pickle.dump((silhouette_scores, davies_bouldin_scores, inertia_values, explained_variances), f)

# Plot variance explained
plt.figure(figsize=(10, 6))
for n_components, cumulative_variance in explained_variances.items():
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', label=f"{n_components} Themes")

plt.axhline(y=0.9, color='r', linestyle='--', label='90% Variance Explained')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained by LSI Components')
plt.legend()
plt.show()

# Plot clustering quality
silhouette_values = [x[1] for x in silhouette_scores]
davies_bouldin_values = [x[1] for x in davies_bouldin_scores]
theme_values = [x[0] for x in silhouette_scores]

plt.figure(figsize=(10, 6))
plt.plot(theme_values, silhouette_values, marker='o', label='Silhouette Score')
plt.plot(theme_values, davies_bouldin_values, marker='o', label='Davies-Bouldin Index')
plt.xlabel('Number of Themes')
plt.ylabel('Score')
plt.title('Clustering Quality vs. Number of Themes')
plt.legend()
plt.show()

# Plot Elbow Method for each number of themes
import pandas as pd

results = pd.DataFrame(inertia_values, columns=['Themes', 'Clusters', 'Inertia'])

plt.figure(figsize=(10, 6))
for n_components in theme_range:
    subset = results[results['Themes'] == n_components]
    plt.plot(subset['Clusters'], subset['Inertia'], marker='o', label=f'{n_components} Themes')

plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Clusters by Themes')
plt.legend()
plt.show()
