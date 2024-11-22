import pickle
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import os

# Load the TF-IDF matrix
with open("pkl_files/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

if not isinstance(tfidf_matrix, csr_matrix):
    raise TypeError("TF-IDF matrix is not in a sparse format. Convert it to CSR or CSC before proceeding.")


n_components = 100  # Define the number of themes
U, Sigma, VT = svds(tfidf_matrix, k=n_components)

# Reverse the order of components (svds gives smallest components first)
U = U[:, ::-1]
Sigma = Sigma[::-1]
VT = VT[::-1, :]

# Compute the LSI matrix (document vectors in latent space)
lsi_matrix = np.dot(U, np.diag(Sigma))

# Access the reduced term-topic matrix for interpretation
lsi_topics = VT

# Save the LSI vectors (document representations) for later use
with open("pkl_files/lsi_vectors.pkl", "wb") as f:
    pickle.dump(lsi_matrix, f)

# Load vocabulary for interpretation
with open("pkl_files/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
vocabulary = vectorizer.get_feature_names_out()

# Extract themes and write to a text file
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "lsi_concepts.txt")

n_top_words = 10  # Number of words to display for each theme
threshold = 0.01  # Define the threshold for "significant" word weights

with open(output_file, "w", encoding="utf-8") as f:
    for i, topic in enumerate(lsi_topics):
        f.write(f"Theme {i+1}:\n")
        
        # Top words for this theme
        top_words = [vocabulary[j] for j in topic.argsort()[-n_top_words:]]
        f.write("Top Words: " + " ".join(top_words) + "\n")
        
        # Count the number of significant words
        significant_word_count = np.sum(np.abs(topic) > threshold)
        f.write(f"Number of significant words: {significant_word_count}\n\n")
