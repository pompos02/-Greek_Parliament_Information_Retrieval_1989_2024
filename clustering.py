import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
import pandas as pd

# Load LSI vectors and speech IDs
with open("pkl_files/lsi_vectors.pkl", "rb") as f:
    lsi_vectors = pickle.load(f)

with open("pkl_files/speech_ids.pkl", "rb") as f:
    speech_ids = pickle.load(f)
    

# Normalize LSI vectors for better clustering
lsi_vectors_normalized = normalize(lsi_vectors)

# Define the number of clusters
n_clusters = 50  # Starting with 20 clusters as per your comment

# Define MiniBatch K-Means parameters
mbk = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
mbk.fit(lsi_vectors_normalized)


# Compute silhouette score on a sample
sample_size = 10000
sil_score = silhouette_score(lsi_vectors_normalized, mbk.labels_, metric='cosine', sample_size=sample_size, random_state=42)
print(f"Silhouette Score={sil_score:.4f}")
