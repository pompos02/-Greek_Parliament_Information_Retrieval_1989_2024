import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# Load LSI vectors and speech IDs
with open("pkl_files/lsi_vectors.pkl", "rb") as f:
    lsi_vectors = pickle.load(f)

with open("pkl_files/speech_ids.pkl", "rb") as f:
    speech_ids = pickle.load(f)

# Optional: Normalize LSI vectors for better clustering
from sklearn.preprocessing import normalize
lsi_vectors_normalized = normalize(lsi_vectors)

# Define the number of clusters
n_clusters = 4  # Start with 20 clusters, adjust based on evaluation

# K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(lsi_vectors_normalized)

# Evaluate clustering quality
sil_score = silhouette_score(lsi_vectors_normalized, labels, metric='cosine')
print(f"Silhouette Score: {sil_score:.2f}")
speech_clusters = {speech_ids[i]: labels[i] for i in range(len(labels))}


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reduce dimensions to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
lsi_2d = tsne.fit_transform(lsi_vectors)

# Plot clusters
plt.scatter(lsi_2d[:, 0], lsi_2d[:, 1], c=labels, cmap='tab10', s=10)
plt.colorbar()
plt.title("K-Means Clustering of Speeches")
plt.show()
