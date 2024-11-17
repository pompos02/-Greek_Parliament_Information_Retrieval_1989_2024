import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import pickle
import nltk
from nltk.corpus import stopwords
import unicodedata
import string
from greek_stemmer import stemmer  # Correctly import the Stemmer class
import spacy
import re

nltk.download('stopwords')
nltk.download('punkt')

# Greek stopwords
greek_stopwords = set(stopwords.words('greek'))

# Load Greek language model in SpaCy
try:
    nlp = spacy.load("el_core_news_sm")
except Exception as e:
    print("Please run this: python -m spacy download el_core_news_sm==3.7.0")
    exit(1)



def remove_accents(text):
    """Remove accents from Greek words."""
    if text is None:
        return ''
    normalized_text = unicodedata.normalize('NFD', text)
    accent_removed_text = ''.join(
        char for char in normalized_text if unicodedata.category(char) != 'Mn'
    )
    return unicodedata.normalize('NFC', accent_removed_text)

def word_stemming(word: str) -> str:
    """Stem a word based on its part of speech."""
    doc = nlp(word)
    tag = doc[0].pos_
    try:
        if tag == "NOUN":
            return stemmer.stem_word(word, "NNM").lower()
        elif tag == "VERB":
            return stemmer.stem_word(word, "VB").lower()
        elif tag == "ADJ" or tag == "ADV":
            return stemmer.stem_word(word, "JJM").lower()
        elif tag == "PROPN":
            return stemmer.stem_word(word, "PRP").lower()
        else:
            return stemmer.stem_word(word, "NNM").lower()
    except Exception:
        print(word)
        print("Something went wrong, moving on to the next word...")
        return ""

def remove_unwanted_pattern(word: str) -> str:
    """Remove unwanted patterns from a word."""
    unwanted_pattern = re.compile(r'[0-9@#$%^&*()-_=+[\]{};:\'",.<>/?\\|`~!]')
    cleaned_word = re.sub(unwanted_pattern, '', word)
    cleaned_word = re.sub(r'\t', '', cleaned_word)
    if (cleaned_word == " ") or (cleaned_word.lower() in greek_stopwords) or (len(cleaned_word) == 1) or (
            cleaned_word == ""):
        cleaned_word = ""
    return cleaned_word


def preprocess_text(text):
    """Preproces s text by applying accent removal, tokenization, stopword removal, and stemming."""
    if text is None:
        return ''
    
    # 1. Remove accents
    text = remove_accents(text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # 4. Remove unwanted patterns and Greek stopwords
    tokens = [remove_unwanted_pattern(token) for token in tokens]
    tokens = [token for token in tokens if token]  # Filter out empty strings
    
    # 5. Apply stemming
    tokens = [word_stemming(token) for token in tokens if token]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Connect to your PostgreSQL database
db_name = 'speeches'
db_user = 'pompos02'  # Replace with your PostgreSQL username
db_password = 'mypassword123'  # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

# Load speeches from the database
df = pd.read_sql('SELECT id, speech FROM speeches', con=engine)

# Apply the preprocessing function
df['preprocessed_speech'] = df['speech'].apply(preprocess_text)

# Initialize TfidfVectorizer using preprocessed text
vectorizer = TfidfVectorizer(
    max_features=10000,         # Limit vocabulary size to top 10,000 terms
    lowercase=False             # Lowercasing already handled in preprocessing
)

tfidf_matrix = vectorizer.fit_transform(df['preprocessed_speech'])

# Save the vectorizer and TF-IDF matrix to disk
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# Optionally, save the speech IDs for reference
with open('speech_ids.pkl', 'wb') as f:
    pickle.dump(df['id'].tolist(), f)

print("TF-IDF vectors computed and saved successfully.")