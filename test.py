from flask import render_template, request, jsonify

from sqlalchemy import text, create_engine
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, normalize
import unicodedata
import re
from nltk.corpus import stopwords
import spacy
from greek_stemmer import stemmer
import pandas as pd

db_name = 'speeches'
db_user = 'pompos02'            # Replace with your PostgreSQL username
db_password = 'mypassword123'   # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')


def load_tfidf_components():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('tfidf_matrix.pkl', 'rb') as f:
            tfidf_matrix = pickle.load(f)
        
        with open('speech_ids.pkl', 'rb') as f:
            speech_ids = pickle.load(f)
            
        return vectorizer, tfidf_matrix, speech_ids
    except Exception as e:
        print(f"Error loading TF-IDF components: {e}")
        return None, None, None

# Initialize components
vectorizer, tfidf_matrix, speech_ids = load_tfidf_components()

# Initialize SpaCy
try:
    nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])
except Exception as e:
    print(f"Error loading SpaCy model: {e}")
    exit(1)

# Initialize other components
greek_stopwords = set(stopwords.words('greek'))
UNWANTED_PATTERN = re.compile(r'[0-9@#$%^&*()\-\_=+\[\]{};:\'",.<>/?\\|`~!]')
TAB_PATTERN = re.compile(r'\t+')

def preprocess_query(query_text):
    """Preprocess the search query using the same pipeline as the documents."""
    def remove_accents(text):
        if not text:
            return ''
        normalized_text = unicodedata.normalize('NFD', text)
        accent_removed_text = ''.join(
            char for char in normalized_text if unicodedata.category(char) != 'Mn'
        )
        return unicodedata.normalize('NFC', accent_removed_text)

    # Process with SpaCy
    doc = nlp(query_text)
    tokens = []
    
    for token in doc:
        cleaned_token = UNWANTED_PATTERN.sub('', token.text)
        cleaned_token = TAB_PATTERN.sub('', cleaned_token)
        
        if (not cleaned_token) or (cleaned_token.lower() in greek_stopwords) or (len(cleaned_token) == 1):
            continue
        
        pos = token.pos_
        try:
            if pos == "NOUN":
                stemmed = stemmer.stem_word(cleaned_token, "NNM").lower()
            elif pos == "VERB":
                stemmed = stemmer.stem_word(cleaned_token, "VB").lower()
            elif pos in {"ADJ", "ADV"}:
                stemmed = stemmer.stem_word(cleaned_token, "JJM").lower()
            elif pos == "PROPN":
                stemmed = stemmer.stem_word(cleaned_token, "PRP").lower()
            else:
                stemmed = stemmer.stem_word(cleaned_token, "NNM").lower()
            
            if stemmed:
                tokens.append(stemmed)
        except Exception as e:
            print(f"Error stemming word '{cleaned_token}': {e}")
            continue
    
    return ' '.join(tokens)

def get_speech_excerpt(speech, max_length=300):
    """Create a relevant excerpt from the speech text"""
    if len(speech) <= max_length:
        return speech
    
    # Get the middle portion of the speech
    start = (len(speech) - max_length) // 2
    excerpt = speech[start:start + max_length]
    
    # Ensure we don't cut words in the middle
    first_space = excerpt.find(' ')
    last_space = excerpt.rfind(' ')
    
    return '...' + excerpt[first_space:last_space].strip() + '...'


query = "Βίως"


try:

    processed_query = preprocess_query(query)
    preprocessed_tokens = processed_query.split()
    vocab = set(vectorizer.vocabulary_.keys())
    tokens_in_vocab = [token for token in preprocessed_tokens if token in vocab]

    


    print("Tokens in query that are in the vocabulary:", tokens_in_vocab)
    print("Sample Vocabulary Terms:", list(vectorizer.vocabulary_.keys())[:50])

    print("original query: ",query)
    
    processed_query = preprocess_query(query)
    print("proccesed query: ",processed_query)
    # Preprocess the query
    # Transform query to TF-IDF vector
    query_vector = vectorizer.transform([processed_query])
    query_vector = normalize(query_vector, norm='l2')
    normalized_tfidf_matrix = normalize(tfidf_matrix, norm='l2')
    print("Query Vector Shape:", query_vector.shape)
    print("Query Vector Data:", query_vector)
    print("Non-zero elements in query vector:", query_vector.nnz)
    # Calculate similarity scores
    similarity_scores = cosine_similarity(query_vector, normalized_tfidf_matrix).flatten()
    print("Similarity scores:", similarity_scores)

# Check if at least one similarity score is non-zero
    if np.any(similarity_scores > 0):
        print("Non-zero similarity scores found.")
    else:
        print("All similarity scores are zero.")
    # Get top matches above threshold
    threshold = 0.1
    top_indices = similarity_scores.argsort()[::-1]
    
    # Filter results above threshold
    print("top indices: ",top_indices)
    term_index = vectorizer.vocabulary_.get("βιως")
    print(f"TF-IDF weight for 'βιως':", tfidf_matrix[:, term_index].toarray())
    
    # Check if at least one similarity score is non-zero
    if np.any(similarity_scores > 0):
        print("Non-zero similarity scores found.")
        
        # Get top matches (sorted in descending order)
        top_indices = similarity_scores.argsort()[::-1]
        
        # Filter results above a certain threshold (optional)
        threshold = 0.1
        top_matches = [(idx, similarity_scores[idx]) for idx in top_indices if similarity_scores[idx] > threshold]
        
        if top_matches:
            # Get the top-ranked match
            top_index, top_similarity = top_matches[0]  # Take the highest similarity score
            print(f"Top-ranked speech index: {top_index}")
            print(f"Top similarity score: {top_similarity}")
            print("Content of speech_ids at top index:", speech_ids[top_index])
            if 0 <= top_index < len(speech_ids):
                speech_id = speech_ids[top_index]
                print(f"Top-ranked speech ID: {speech_id}")
                speech_query = 'SELECT speech FROM speeches WHERE id = %s'  # Use %s as the placeholder
                speech_record = pd.read_sql_query(speech_query, con=engine, params=(speech_id,))
                if not speech_record.empty:
                    print("Top-ranked speech content:", speech_record['speech'].iloc[0])
                else:
                    print("No speech found for the given ID.")

            else:
                print(f"Index {top_index} is out of bounds.")


            # Assuming speech_ids is a list or array with corresponding speech texts
            if 0 <= top_index < len(speech_ids):
                print(f"Top-ranked speech (ID: {speech_ids[top_index]}):")
                # Fetch and display the speech text (or excerpt, if applicable)
                print(get_speech_excerpt(speech_ids[top_index]))
            else:
                print("Top index out of bounds for speech_ids.")
        else:
            print("No matches above the similarity threshold.")
    else:
        print("All similarity scores are zero.")

    
    # Fetch speeches from database using SQLAlchemy
    

    
    # Calculate pagination info

    
except Exception as e:
    print(f"Search error: {e}")
    