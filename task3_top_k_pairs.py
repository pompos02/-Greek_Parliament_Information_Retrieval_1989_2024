import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


def load_pickle_files():
    """Load the precomputed pickle files."""
    with open('tfidf_vectors/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    with open('tfidf_vectors/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    with open('tfidf_vectors/speech_ids.pkl', 'rb') as f:
        speech_ids = pickle.load(f)

    return vectorizer, tfidf_matrix, speech_ids


def get_top_k_similar_pairs(cosine_sim_matrix, speech_ids, k=10):
    """Get the top-k pairs with the highest cosine similarity scores."""
    # Create a DataFrame to hold the cosine similarity values
    similarity_df = pd.DataFrame(cosine_sim_matrix, index=speech_ids, columns=speech_ids)

    # Set the diagonal to NaN to exclude self-similarities
    np.fill_diagonal(cosine_sim_matrix, np.nan)

    # Unstack the matrix to get a series of pairwise similarities
    similarity_series = similarity_df.unstack()

    # Remove duplicate pairs by sorting indices within each pair
    similarity_series.index = similarity_series.index.map(lambda x: tuple(sorted(x)))
    similarity_series = similarity_series[~similarity_series.index.duplicated()]

    # Sort the similarities in descending order
    sorted_similarities = similarity_series.sort_values(ascending=False)

    top_k_pairs = sorted_similarities.head(k)

    return top_k_pairs


def main(k):
    vectorizer, tfidf_matrix, speech_ids = load_pickle_files()

    cosine_sim_matrix = cosine_similarity(tfidf_matrix)  # cosine similarity for all members (NxN matrix)

    top_k_pairs = get_top_k_similar_pairs(cosine_sim_matrix, speech_ids, k)

    print(f"Top {k} most similar pairs:")
    for idx, similarity in top_k_pairs.items():
        member1, member2 = idx
        print(f"Members: {member1} - {member2}, Similarity: {similarity:.4f}")


if __name__ == "__main__":
    main(k=10)
