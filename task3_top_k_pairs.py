import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sqlalchemy import text
from db import get_db


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


def get_political_parties(member_name):
    """Fetch political parties for a given member name from the database."""
    sql_query = text("""
        SELECT DISTINCT political_party 
        FROM merged_speeches
        WHERE member_name = :member_name
    """)
    engine = get_db()
    with engine.connect() as conn:
        result = conn.execute(sql_query, {'member_name': member_name})
        # Access the first element of the tuple for 'political_party'
        parties = [row[0] for row in result]
    return ', '.join(parties) if parties else "Unknown"


# def main(k):
#     vectorizer, tfidf_matrix, speech_ids = load_pickle_files()
#
#     cosine_sim_matrix = cosine_similarity(tfidf_matrix)  # cosine similarity for all members (NxN matrix)
#
#     top_k_pairs = get_top_k_similar_pairs(cosine_sim_matrix, speech_ids, k)
#
#     print(f"Top {k} most similar pairs:\n")
#     for i, ((member1, member2), similarity) in enumerate(top_k_pairs.items(), start=1):
#         # Fetch political parties for both members
#         print(member1)
#         member1_parties = get_political_parties(member1)
#         member2_parties = get_political_parties(member2)
#
#         # Print result with numbering
#         print(f"{i}. {member1} ({member1_parties}) - {member2} ({member2_parties}), Similarity: {similarity:.4f}")
#
#
# if __name__ == "__main__":
#     main(k=10)
