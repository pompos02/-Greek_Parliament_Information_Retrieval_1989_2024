import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import joblib
import pandas as pd

# Load LSI vectors and speech IDs
with open("pkl_files/lsi_vectors.pkl", "rb") as f:
    lsi_vectors = pickle.load(f)

with open("pkl_files/speech_ids.pkl", "rb") as f:
    speech_ids = pickle.load(f)

# Normalize LSI vectors for better clustering
lsi_vectors_normalized = normalize(lsi_vectors)

# Define the number of clusters
n_clusters = 20  # Starting with 20 clusters as per your comment

# Define MiniBatch K-Means parameters
batch_size = 1000  # Adjust based on your system's memory and dataset size
max_iterations = 100  # Number of iterations (epochs) over the dataset

# Initialize MiniBatchKMeans
mbk = MiniBatchKMeans(
    n_clusters=n_clusters,
    batch_size=batch_size,
    max_iter=max_iterations,
    random_state=42,
    init='k-means++',
    reassignment_ratio=0.01
)

# Fit and predict cluster labels
labels = mbk.fit_predict(lsi_vectors_normalized)

# Evaluate clustering quality
# For large datasets, it's computationally expensive to compute silhouette score for all samples
# Therefore, we use a sample
sample_size = 10000  # Adjust based on available memory and desired accuracy
sil_score = silhouette_score(lsi_vectors_normalized, labels, metric='cosine', sample_size=sample_size, random_state=42)
print(f"Silhouette Score: {sil_score:.2f}")


# Save cluster labels along with speech IDs
clustered_speeches = pd.DataFrame({
    'speech_id': speech_ids,
    'cluster_label': labels
})

clustered_speeches.to_csv('clustered_speeches.csv', index=False)
