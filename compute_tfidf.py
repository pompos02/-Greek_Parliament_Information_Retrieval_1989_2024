import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import pickle
import nltk
from nltk.corpus import stopwords
import unicodedata
import re
from greek_stemmer import stemmer  # Ensure this is correctly installed and imported
import spacy
from tqdm import tqdm
from functools import partial
import multiprocessing

# Download necessary NLTK data


#nltk.download('stopwords')
#nltk.download('punkt')

# Initialize tqdm for pandas
tqdm.pandas()

# Greek stopwords
greek_stopwords = set(stopwords.words('greek'))

# Precompile regex patterns for performance
UNWANTED_PATTERN = re.compile(r'[0-9@#$%^&*()\-\_=+\[\]{};:\'",.<>/?\\|`~!]')
TAB_PATTERN = re.compile(r'\t+')

# Initialize SpaCy with multiple cores for parallel processing
try:
    # Load SpaCy model with disabled components to speed up processing
    nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])
except Exception as e:
    print("Please run this: python -m spacy download el_core_news_sm==3.7.0")
    exit(1)

def remove_accents(text):
    """Remove accents from Greek words."""
    if not text:
        return ''
    normalized_text = unicodedata.normalize('NFD', text)
    accent_removed_text = ''.join(
        char for char in normalized_text if unicodedata.category(char) != 'Mn'
    )
    return unicodedata.normalize('NFC', accent_removed_text)

def preprocess_documents(texts, batch_size=500, n_process=4):
    """
    Preprocess a list of texts by removing accents, lowercasing, tokenizing,
    removing unwanted patterns and stopwords, and applying stemming.
    
    Args:
        texts (iterable): An iterable of text strings.
        batch_size (int): Number of texts to process in each batch.
        n_process (int): Number of worker processes for SpaCy.
        
    Returns:
        list: A list of preprocessed text strings.
    """
    preprocessed_texts = []
    
    # Use nlp.pipe for efficient batch processing with progress bar
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), 
                   total=len(texts), desc="Preprocessing"):
        tokens = []
        for token in doc:
            # Remove unwanted patterns
            cleaned_token = UNWANTED_PATTERN.sub('', token.text)
            cleaned_token = TAB_PATTERN.sub('', cleaned_token)
            
            # Skip if token is empty, a stopword, or a single character
            if (not cleaned_token) or (cleaned_token.lower() in greek_stopwords) or (len(cleaned_token) == 1):
                continue
            
            # Apply stemming based on POS tag
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
        
        preprocessed_text = ' '.join(tokens)
        preprocessed_texts.append(preprocessed_text)
    
    return preprocessed_texts

def main():
    # Database connection parameters
    db_name = 'speeches'
    db_user = 'pompos02'        # Replace with your PostgreSQL username
    db_password = 'mypassword123'  # Replace with your PostgreSQL password
    db_host = 'localhost'
    db_port = '5432'
    engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')
    
    # Load speeches from the database
    print("Loading data from the database...")
    df = pd.read_sql('SELECT id, speech FROM speeches', con=engine)
    
    # Preprocess texts with progress bar
    print("Preprocessing texts...")
    df['preprocessed_speech'] = preprocess_documents(
        df['speech'].tolist(),
        batch_size=1000,   # Adjust based on your system's memory
        n_process=multiprocessing.cpu_count() - 1  # Utilize available CPU cores
    )
    
    # Initialize TfidfVectorizer
    print("Initializing TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size to top 10,000 terms
        lowercase=False      # Lowercasing already handled in preprocessing
    )
    
    # Fit and transform the data with progress bar
    print("Fitting and transforming TF-IDF matrix...")
    tfidf_matrix = vectorizer.fit_transform(
        tqdm(df['preprocessed_speech'], desc="TF-IDF", total=df.shape[0])
    )
    
    # Save the vectorizer and TF-IDF matrix to disk
    print("Saving the vectorizer and TF-IDF matrix...")
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    
    # Optionally, save the speech IDs for reference
    with open('speech_ids.pkl', 'wb') as f:
        pickle.dump(df['id'].tolist(), f)
    
    print("TF-IDF vectors computed and saved successfully.")

if __name__ == "__main__":
    main()
