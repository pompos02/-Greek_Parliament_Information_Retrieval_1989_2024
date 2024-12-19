import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import pandas as pd
from db import get_db
import numpy as np
from sqlalchemy.sql import text
import matplotlib.pyplot as plt
import seaborn as sns
from umap import UMAP

# Load LSI vectors and speech IDs
with open("pkl_files/lsi_vectors.pkl", "rb") as f:
    lsi_vectors = pickle.load(f)

with open("pkl_files/speech_ids.pkl", "rb") as f:
    speech_ids = pickle.load(f)

# Normalize LSI vectors for better clustering
lsi_vectors_normalized = normalize(lsi_vectors)

# Path to save and load the KMeans labels
labels_file = "pkl_files/Kmeans_labels.pkl"

# Define K-Means parameters
n_clusters = 50
mbk = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
mbk.fit(lsi_vectors_normalized)

# Save the labels and cluster centers to a pickle file
clusters = mbk.labels_
cluster_centers = mbk.cluster_centers_

# compute silhouette score
sample_size = 10000
sil_score = silhouette_score(lsi_vectors_normalized, clusters, metric='cosine', sample_size=sample_size,
                             random_state=42)
print(f"Silhouette Score={sil_score:.4f}")

engine = get_db()

df = pd.DataFrame({
    'speech_id': speech_ids,
    'cluster': clusters
})

best_speeches_per_cluster = {}

# Loop through each cluster and find the closest 2 speeches to the cluster center
for cluster in np.unique(clusters):
    # Get the indices of the speeches in the current cluster
    cluster_indices = df[df['cluster'] == cluster].index

    # Get the cluster center
    cluster_center = mbk.cluster_centers_[cluster]

    # Calculate distances from the cluster center (using cosine distance)
    distances = np.linalg.norm(lsi_vectors_normalized[cluster_indices] - cluster_center, axis=1)

    # Find the indices of the two smallest distances (best 2 speeches)
    best_2_indices = cluster_indices[np.argsort(distances)[:2]]

    # Store the best 2 speeches' IDs in the dictionary
    best_speeches_per_cluster[cluster] = df.loc[best_2_indices, 'speech_id'].values

query = text("""
    SELECT id, member_name, sitting_date, political_party,roles, merged_speech 
    FROM merged_speeches 
    WHERE id IN :speech_ids
""")
# Flatten the speech IDs from the dictionary and convert to int
speech_ids_to_fetch = [int(id) for cluster_ids in best_speeches_per_cluster.values() for id in cluster_ids]

with engine.connect() as connection:
    result = connection.execute(query, {'speech_ids': tuple(speech_ids_to_fetch)})

# Convert result to a DataFrame
speech_details = pd.DataFrame(result.fetchall(),
                              columns=['id', 'member_name', 'sitting_date', 'political_party', 'roles', 'speech'])

# Write results to a text file
output_file = "output/best_speeches_per_cluster_lsi.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for cluster, speech_ids in best_speeches_per_cluster.items():
        file.write(f"Cluster {cluster}:\n")
        for speech_id in speech_ids:
            speech_data = speech_details[speech_details['id'] == speech_id].iloc[0]
            file.write(f"  Speech ID: {speech_data['id']}\n")
            file.write(f"  Member Name: {speech_data['member_name']}\n")
            file.write(f"  Sitting Date: {speech_data['sitting_date']}\n")
            file.write(f"  Political Party: {speech_data['political_party']}\n")
            file.write(f"   Role: {speech_data['roles']}\n")
            file.write(f"  Speech: {speech_data['speech'][:200]}...\n")  # First 200 chars for brevity
            file.write("\n")
        file.write("\n")

print(f"Results saved to {output_file}")

# Reduce dimensions to 2D using UMAP
umap_reducer_2d = UMAP(n_components=2, random_state=42)
lsi_vectors_2d = umap_reducer_2d.fit_transform(lsi_vectors_normalized)

# Create a DataFrame for visualization
visualization_df_2d = pd.DataFrame({
    'x': lsi_vectors_2d[:, 0],
    'y': lsi_vectors_2d[:, 1],
    'cluster': clusters
})

plt.figure(figsize=(12, 8))

# Generate a color palette
palette = sns.color_palette("tab20", n_colors=n_clusters)

# Plot each cluster with a unique color
sns.scatterplot(
    x='x',
    y='y',
    hue='cluster',
    palette=palette,
    data=visualization_df_2d,
    legend='full',
    alpha=0.7
)

plt.title("K-Means Clusters Visualized with UMAP (2D)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

# Adjust legend and show the plot
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
