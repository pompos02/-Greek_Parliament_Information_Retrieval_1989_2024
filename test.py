import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, Table, MetaData
import matplotlib.pyplot as plt
from db import get_db

# Load TF-IDF Matrix and Feature Names
def load_tfidf_data():
    # Load the TF-IDF matrix
    with open('pkl_files/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Load the vectorizer to get feature names
    with open('pkl_files/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names



def get_top_keywords(tfidf_matrix, feature_names, top_n=10):
    keywords = []
    for i in range(tfidf_matrix.shape[0]):  # Iterate over rows
        row = tfidf_matrix[i].toarray().flatten()  # Convert sparse row to dense
        indices = np.argsort(row)[::-1][:top_n]  # Sort indices by TF-IDF score
        keywords.append([(feature_names[idx], row[idx]) for idx in indices])
    return keywords


# Connect to PostgreSQL Database
def connect_to_db():
    engine = get_db
    metadata = MetaData()
    merged_speeches = Table("merged_speeches", metadata, autoload_with=engine)
    return engine, merged_speeches

if __name__ == "__main__":
    engine, merged_speeches = connect_to_db()
    print("Connection successful!")
    print(merged_speeches.columns.keys())  # Should display column names of `merged_speeches`
