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
with open("pkl_files/nmf_vectors.pkl", "rb") as f:
    nmf_vectors = pickle.load(f)

with open("pkl_files/speech_ids.pkl", "rb") as f:
    speech_ids = pickle.load(f)

# Normalize LSI vectors for better clustering
nmf_vectors_normalized = normalize(nmf_vectors)

# Path to save and load the KMeans labels
labels_file = "pkl_files/Kmeans_labels.pkl"


# Define K-Means parameters
n_clusters = 70
mbk = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
mbk.fit(nmf_vectors_normalized)

# Save the labels and cluster centers to a pickle file
clusters = mbk.labels_
cluster_centers = mbk.cluster_centers_

# Optionally compute silhouette score
sample_size = 10000
sil_score = silhouette_score(nmf_vectors_normalized, clusters, metric='cosine', sample_size=sample_size, random_state=42)
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
    distances = np.linalg.norm(nmf_vectors_normalized[cluster_indices] - cluster_center, axis=1)
    
    # Find the indices of the two smallest distances (best 2 speeches)
    best_2_indices = cluster_indices[np.argsort(distances)[:2]]
    
    # Store the best 2 speeches' IDs in the dictionary
    best_speeches_per_cluster[cluster] = df.loc[best_2_indices, 'speech_id'].values


# Assuming you have a 'speeches' table with an 'id' column
query = text("""
    SELECT id, member_name, sitting_date, political_party,roles, merged_speech 
    FROM merged_speeches 
    WHERE id IN :speech_ids
""")
# Flatten the speech IDs from the dictionary and pass them to the query
# Flatten the speech IDs from the dictionary and convert to Python int
speech_ids_to_fetch = [int(id) for cluster_ids in best_speeches_per_cluster.values() for id in cluster_ids]


with engine.connect() as connection:
    result = connection.execute(query, {'speech_ids': tuple(speech_ids_to_fetch)})

# Convert result to a DataFrame
speech_details = pd.DataFrame(result.fetchall(), columns=['id', 'member_name', 'sitting_date','political_party','roles','speech'])

# Write results to a text file
output_file = "output/best_speeches_per_cluster_nmf.txt"
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

umap_reducer = UMAP(n_components=3, random_state=42)
lsi_vectors_3d = umap_reducer.fit_transform(nmf_vectors_normalized)

# Create a DataFrame for visualization
visualization_df = pd.DataFrame({
    'x': lsi_vectors_3d[:, 0],
    'y': lsi_vectors_3d[:, 1],
    'z': lsi_vectors_3d[:, 2],
    'cluster': clusters
})

# Plot the 3D clusters
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate a color palette
palette = sns.color_palette("tab20", n_colors=n_clusters)
colors = [palette[cluster] for cluster in clusters]

# Plot the points
scatter = ax.scatter(
    visualization_df['x'],
    visualization_df['y'],
    visualization_df['z'],
    c=colors,
    s=10,  # Marker size
    alpha=0.7  # Transparency
)

# Set plot labels and title
ax.set_title("K-Means Clusters Visualized with UMAP (3D)")
ax.set_xlabel("UMAP Dimension 1")
ax.set_ylabel("UMAP Dimension 2")
ax.set_zlabel("UMAP Dimension 3")

# Show the plot
plt.show()