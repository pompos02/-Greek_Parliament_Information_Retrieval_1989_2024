import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.corpus import stopwords
import re
from greek_stemmer import stemmer
import spacy
import os
from db import get_db


def preprocess_text(text, nlp, greek_stopwords, UNWANTED_PATTERN, TAB_PATTERN, stemmer):
    """
    Preprocess a single text.
    """
    doc = nlp(text)
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

    return ' '.join(tokens)


def split_text(text, chunk_size=2000000):
    """Splits a long text into smaller chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def preprocess_speech(text, nlp, greek_stopwords, UNWANTED_PATTERN, TAB_PATTERN, stemmer):
    """Preprocess long speeches by splitting them into smaller chunks."""
    if len(text) > nlp.max_length:
        chunks = split_text(text)
        processed_chunks = [preprocess_text(chunk, nlp, greek_stopwords, UNWANTED_PATTERN, TAB_PATTERN, stemmer) for
                            chunk in chunks]
        return ' '.join(processed_chunks)  # Combine the processed chunks back together
    else:
        return preprocess_text(text, nlp, greek_stopwords, UNWANTED_PATTERN, TAB_PATTERN, stemmer)


def main():
    # Greek stopwords
    greek_stopwords = set(stopwords.words('greek'))

    # Precompile regex patterns for performance
    UNWANTED_PATTERN = re.compile(r'[0-9@#$%^&*()\-\_=+\[\]{};:\'",.<>/?\\|`~!]')
    TAB_PATTERN = re.compile(r'\t+')

    # Initialize SpaCy
    try:
        nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])
        nlp.max_length = 4000000
    except Exception as e:
        print("Please run this: python -m spacy download el_core_news_sm")
        exit(1)

    engine = get_db()

    print("Loading and grouping data by member_name...")
    query = """
    SELECT
        member_name,
        STRING_AGG(merged_speech, ' ') AS merged_speech
    FROM
        merged_speeches
    GROUP BY
        member_name
    """
    grouped_speeches = pd.read_sql_query(query, con=engine)

    print("Preprocessing speeches...")
    grouped_speeches['preprocessed_speech'] = grouped_speeches['merged_speech'].apply(
        lambda x: preprocess_speech(
            x,
            nlp=nlp,
            greek_stopwords=greek_stopwords,
            UNWANTED_PATTERN=UNWANTED_PATTERN,
            TAB_PATTERN=TAB_PATTERN,
            stemmer=stemmer
        )
    )

    print("Initializing TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        lowercase=False
    )

    # Prepare data for TF-IDF processing
    speeches = grouped_speeches['preprocessed_speech'].tolist()
    speech_ids = grouped_speeches['member_name'].tolist()  # Using member_name as speech ID for simplicity

    print("Computing TF-IDF matrix...")
    tfidf_matrix = vectorizer.fit_transform(speeches)

    tfidf_dir = 'pkl_files/top_k_pairs_vectors'
    os.makedirs(tfidf_dir, exist_ok=True)

    print("Saving the TF-IDF matrix...")
    with open(os.path.join(tfidf_dir, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    print("Saving the speech IDs...")
    with open(os.path.join(tfidf_dir, 'speech_ids.pkl'), 'wb') as f:
        pickle.dump(speech_ids, f)

    print("TF-IDF vectors computed and saved successfully.")


if __name__ == "__main__":
    main()
