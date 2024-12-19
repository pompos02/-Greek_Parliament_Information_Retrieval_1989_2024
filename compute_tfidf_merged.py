import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from nltk.corpus import stopwords
import re
from greek_stemmer import stemmer  # Ensure this is correctly installed and imported
import spacy
from tqdm import tqdm
import os
from db import get_db


def preprocess_documents_chunk(texts, nlp, greek_stopwords, UNWANTED_PATTERN, TAB_PATTERN, stemmer, batch_size,
                               n_process=1):
    """
    Generator that preprocesses texts in chunks to manage memory usage.
    """
    for doc in nlp.pipe(texts, batch_size=batch_size, n_process=n_process, disable=["parser", "ner"]):
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
        yield preprocessed_text


def main():
    # Download necessary NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')

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
        print("Please run this: python -m spacy download el_core_news_sm")
        exit(1)

    # Connect to the PostgreSQL database
    engine = get_db()

    # Count total records to set tqdm's total parameter
    count_query = 'SELECT COUNT(*) FROM merged_speeches'
    total_records_df = pd.read_sql_query(count_query, con=engine)
    total_records = total_records_df.iloc[0, 0]  # 1,354,685

    chunksize = 100

    # Load speeches from the database in chunks
    print("Loading data from the database...")
    query = 'SELECT id, merged_speech FROM merged_speeches ORDER BY sitting_date ASC'

    # Initialize TfidfVectorizer
    print("Initializing TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Limit vocabulary size to top 10,000 terms
        lowercase=False  # Lowercasing already handled in preprocessing
    )

    # Directory to save preprocessed chunks
    preprocessed_dir = 'preprocessed_chunks'
    os.makedirs(preprocessed_dir, exist_ok=True)

    # Initialize tqdm progress bar
    print("Preprocessing texts...")
    pbar = tqdm(total=total_records, desc="Preprocessing", unit="rows")

    reader = pd.read_sql_query(query, con=engine, chunksize=chunksize)
    chunk_number = 0
    for chunk in reader:
        # Remove rows where 'speech' is None or NaN
        chunk = chunk.dropna(subset=['merged_speech'])
        # Extract 'id's and 'speech's
        ids = chunk['id'].tolist()
        speeches = chunk['merged_speech'].tolist()

        # Preprocess the current chunk
        chunk_preprocessed = list(preprocess_documents_chunk(
            texts=speeches,
            nlp=nlp,
            greek_stopwords=greek_stopwords,
            UNWANTED_PATTERN=UNWANTED_PATTERN,
            TAB_PATTERN=TAB_PATTERN,
            stemmer=stemmer,
            batch_size=25,
            n_process=2
        ))

        # Create a DataFrame for the preprocessed chunk
        preprocessed_df = pd.DataFrame({
            'id': ids,
            'preprocessed_speech': chunk_preprocessed
        })

        # Save to disk (using pickle)
        preprocessed_df.to_pickle(os.path.join(preprocessed_dir, f'preprocessed_chunk_{chunk_number}.pkl'))
        # Update the progress bar by the number of rows processed in this chunk
        pbar.update(len(chunk))

        chunk_number += 1

        # Free memory by deleting the chunk
        del chunk, chunk_preprocessed, preprocessed_df

    pbar.close()

    # After preprocessing, load all preprocessed chunks incrementally for TF-IDF
    print("Loading preprocessed data for TF-IDF...")
    preprocessed_speeches = []
    speech_ids = []
    for i in tqdm(range(chunk_number), desc="Loading Preprocessed Chunks"):
        preprocessed_df = pd.read_pickle(os.path.join(preprocessed_dir, f'preprocessed_chunk_{i}.pkl'))
        speech_ids.extend(preprocessed_df['id'].tolist())
        preprocessed_speeches.extend(preprocessed_df['preprocessed_speech'].tolist())
        del preprocessed_df

    # Create the final DataFrame
    df = pd.DataFrame({
        'id': speech_ids,
        'preprocessed_speech': preprocessed_speeches
    })

    # Fit and transform the data
    print("Fitting and transforming TF-IDF matrix...")
    tfidf_pbar = tqdm(total=len(df['preprocessed_speech']), desc="TF-IDF", unit="rows")
    tfidf_matrix = vectorizer.fit_transform(
        (speech for speech in df['preprocessed_speech'])
    )
    tfidf_pbar.update(len(df['preprocessed_speech']))
    tfidf_pbar.close()

    # Save the vectorizer and TF-IDF matrix to disk
    print("Saving the vectorizer and TF-IDF matrix...")
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

    with open('tfidf_matrix.pkl', 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    # Save the speech IDs for reference
    with open('speech_ids.pkl', 'wb') as f:
        pickle.dump(df['id'].tolist(), f)

    print("TF-IDF vectors computed and saved successfully.")


if __name__ == "__main__":
    main()
