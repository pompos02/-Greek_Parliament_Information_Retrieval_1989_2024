import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import os
from tqdm import tqdm  # Import tqdm
import sys

# Redirect stderr to stdout to capture verbose output
sys.stderr = sys.stdout

# Function to print messages immediately
def print_flush(message):
    print(message, flush=True)

# Load the TF-IDF matrix
print_flush("Loading the TF-IDF matrix...")
with open("pkl_files/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

if not isinstance(tfidf_matrix, csr_matrix):
    raise TypeError("TF-IDF matrix is not in a sparse format. Convert it to CSR or CSC before proceeding.")

# Define the number of components
n_components = 100  # Number of themes

print_flush("Initializing the NMF model...")
# Initialize and fit the NMF model with verbose for progress updates
nmf_model = NMF(
    n_components=n_components,
    init='nndsvda',
    random_state=42,
    max_iter=100,
    solver='mu',
    beta_loss='kullback-leibler',
    verbose=2  # Increase verbosity for more detailed progress
)

print_flush("Fitting the NMF model. This may take some time...")
# Fit the model
try:
    lsi_matrix_nmf = nmf_model.fit_transform(tfidf_matrix)
    nmf_topics = nmf_model.components_
except Exception as e:
    print_flush(f"An error occurred during NMF fitting: {e}")
    sys.exit(1)

print_flush("Saving the NMF vectors...")
# Save the NMF vectors (document representations) for later use
with open("pkl_files/nmf_vectors.pkl", "wb") as f:
    pickle.dump(lsi_matrix_nmf, f)

print_flush("Loading the TF-IDF vectorizer vocabulary...")
# Load vocabulary for interpretation
with open("pkl_files/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
vocabulary = vectorizer.get_feature_names_out()

print_flush("Extracting themes and writing to the output file...")
# Extract themes and write to a text file with a progress bar
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, "nmf_concepts.txt")

n_top_words = 10  # Number of words to display for each theme
threshold = 0.01  # Define the threshold for "significant" word weights

with open(output_file, "w", encoding="utf-8") as f_out:
    for i, topic in tqdm(enumerate(nmf_topics), total=len(nmf_topics), desc="Processing Themes"):
        f_out.write(f"Theme {i+1}:\n")
        
        # Top words for this theme
        top_words_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [vocabulary[j] for j in top_words_indices]
        f_out.write("Top Words: " + " ".join(top_words) + "\n")
        
        # Count the number of significant words
        significant_word_count = np.sum(topic > threshold)
        f_out.write(f"Number of significant words: {significant_word_count}\n\n")

print_flush("NMF processing and theme extraction completed successfully.")
