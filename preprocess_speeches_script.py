import pandas as pd
import spacy
import pickle
import logging
import os
from db import get_db
from greek_stemmer import stemmer
from nltk.corpus import stopwords
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Define stopwords
greek_stopwords = set(stopwords.words('greek'))

# Define unwanted patterns
UNWANTED_PATTERN = re.compile(r'[0-9@#$%^&*()\-\_=+\[\]{};:\'",.<>/?\\|`~!]')
TAB_PATTERN = re.compile(r'\t+')


# Preprocess each speech
def preprocess_speech(text, nlp):
    if not isinstance(text, str):
        logging.warning('Non-string text encountered.')
        return ''

    try:
        doc = nlp(text)
        tokens = []
        for token in doc:
            # Remove unwanted patterns
            cleaned_token = UNWANTED_PATTERN.sub('', token.text)
            cleaned_token = TAB_PATTERN.sub('', cleaned_token)

            # Skip empty, stopword, or single-character tokens
            if not cleaned_token or cleaned_token.lower() in greek_stopwords or len(cleaned_token) == 1:
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
                logging.error(f"Error stemming word '{cleaned_token}': {e}")
                continue

        return ' '.join(tokens)
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return ''


# Main script
if __name__ == "__main__":
    # Establish database connection
    engine = get_db()
    logging.info('Database connection established.')

    # Count total speeches for progress tracking
    total_speeches_query = 'SELECT COUNT(*) FROM merged_speeches'
    total_speeches = pd.read_sql(total_speeches_query, engine).iloc[0, 0]
    logging.info(f'Total speeches in database: {total_speeches}')

    # Directory to save preprocessed speeches
    output_dir = 'pkl_files/preprocess_pkl_files'
    os.makedirs(output_dir, exist_ok=True)

    # Load Greek language model
    try:
        nlp = spacy.load('el_core_news_sm')
        logging.info("SpaCy Greek model 'el_core_news_sm' loaded successfully.")
    except OSError:
        logging.error("SpaCy model 'el_core_news_sm' not found. Please install it using:")
        logging.error("python -m spacy download el_core_news_sm")
        raise

    # Load speeches in chunks
    chunksize = 500  # Adjust as needed
    query = 'SELECT id, sitting_date, political_party, merged_speech FROM merged_speeches'
    reader = pd.read_sql(query, engine, chunksize=chunksize)

    for i, chunk in enumerate(reader):
        logging.info(f'Processing chunk {i + 1}')

        # Preprocess the speeches
        chunk['preprocessed_speech'] = chunk['merged_speech'].apply(lambda x: preprocess_speech(x, nlp))

        # Save the chunk to a pickle file
        chunk_path = os.path.join(output_dir, f'preprocessed_chunk_{i}.pkl')
        with open(chunk_path, 'wb') as f:
            # Include 'id', 'political_party', and 'preprocessed_speech'
            pickle.dump(chunk[['id', 'sitting_date', 'political_party', 'preprocessed_speech']], f)
        logging.info(f'Saved preprocessed chunk {i + 1} to {chunk_path}')

    logging.info('All speeches preprocessed and saved.')

input_dir = 'pkl_files/preprocess_pkl_files'

# Initialize an empty list to hold DataFrames
df_list = []

# Iterate over all pickle files in the directory
for filename in sorted(os.listdir(input_dir)):
    if filename.endswith('.pkl'):
        file_path = os.path.join(input_dir, filename)
        try:
            with open(file_path, 'rb') as f:
                df_chunk = pickle.load(f)
                df_list.append(df_chunk)
            logging.info(f'Loaded {filename} successfully.')
        except Exception as e:
            logging.error(f'Error loading {filename}: {e}')

# Concatenate all DataFrames if any were loaded
if df_list:
    try:
        full_df = pd.concat(df_list, ignore_index=True)
        logging.info('All chunks have been concatenated successfully.')

        # Save the merged DataFrame to a new pickle file
        merged_pickle_path = os.path.join(input_dir, 'preprocessed_all_speeches.pkl')
        with open(merged_pickle_path, 'wb') as f:
            pickle.dump(full_df, f)
        logging.info(f'Merged DataFrame saved to {merged_pickle_path}')
    except Exception as e:
        logging.error(f'Error concatenating DataFrames: {e}')
else:
    logging.warning('No pickle files were found to merge.')
