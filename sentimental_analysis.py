import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import unicodedata
from db import get_db
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Function to remove accents from Greek text
def remove_accents(text):
    if not isinstance(text, str):
        return ''
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

# Establish database connection
engine = get_db()
logging.info('Database connection established.')

# Load the data into a DataFrame
df = pd.read_sql(
    "SELECT * FROM merged_speeches WHERE political_party = 'λαικος συνδεσμος - χρυση αυγη'",
    engine
)
logging.info(f'Data loaded with {len(df)} records.')

# Load the lexicon with all happiness columns
lexicon = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')
logging.info(f'Lexicon loaded with {len(lexicon)} records.')

# Preprocess the lexicon: select relevant columns
happiness_columns = ['Happiness1', 'Happiness2', 'Happiness3', 'Happiness4']
lexicon = lexicon[['Term'] + happiness_columns + ['Comments1']]
logging.info('Selected relevant columns from lexicon.')

# Drop rows with missing Term
lexicon = lexicon.dropna(subset=['Term'])
logging.info(f'Lexicon after dropping rows with missing Term: {len(lexicon)}.')

# Remove accentuation from terms
lexicon['Term'] = lexicon['Term'].apply(remove_accents)
logging.info('Accentuation removed from lexicon terms.')

# Calculate the average happiness score across available columns
lexicon['average_happiness'] = lexicon[happiness_columns].mean(axis=1, skipna=True)

# Drop rows where average_happiness is NaN (i.e., all columns were undefined)
lexicon = lexicon.dropna(subset=['average_happiness'])
logging.info(f'Lexicon after calculating average happiness scores: {len(lexicon)}.')

# Create happiness dictionary
happiness_dict = dict(zip(lexicon['Term'], lexicon['average_happiness']))
logging.info(f'Happiness dictionary created with {len(happiness_dict)} terms.')

# Load Greek language model
try:
    nlp = spacy.load('el_core_news_sm')
    logging.info("SpaCy Greek model 'el_core_news_sm' loaded successfully.")
except OSError:
    logging.error("SpaCy model 'el_core_news_sm' not found. Please install it using:")
    logging.error("python -m spacy download el_core_news_sm")
    raise

# Function to preprocess text: tokenize, lemmatize, remove punctuation and spaces, and remove accents
def preprocess_text(text):
    if not isinstance(text, str):
        logging.warning('Non-string text encountered.')
        return []
    try:
        doc = nlp(text)
        tokens = [
            remove_accents(token.lemma_.lower()) 
            for token in doc 
            if not token.is_punct and not token.is_space
        ]
        return tokens
    except Exception as e:
        logging.error(f"Error processing text: {e}")
        return []

# Function to calculate happiness score
def calculate_happiness(tokens, happiness_dict):
    score = 0
    count = 0
    for token in tokens:
        if token in happiness_dict:
            score += happiness_dict[token]
            count += 1
    if count == 0:
        return 0
    return score / count

# Apply preprocessing and happiness calculation
logging.info('Starting text preprocessing and happiness calculation...')
df['tokens'] = df['merged_speech'].apply(preprocess_text)
df['happiness_score'] = df['tokens'].apply(lambda tokens: calculate_happiness(tokens, happiness_dict))
logging.info('Text preprocessing and happiness calculation completed.')

# Ensure 'sitting_date' is in datetime format
df['sitting_date'] = pd.to_datetime(df['sitting_date'], errors='coerce')

# Extract the year from 'sitting_date'
df['year'] = df['sitting_date'].dt.year

# Drop rows with missing or invalid years
df = df.dropna(subset=['year'])

# Group by year and calculate average happiness score
happiness_trend = df.groupby('year')['happiness_score'].mean().reset_index()

# Visualize the happiness trend over the years
plt.figure(figsize=(12, 6))
sns.lineplot(data=happiness_trend, x='year', y='happiness_score', marker='o')
plt.title('Happiness Trend Over the Years', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Happiness Score', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
