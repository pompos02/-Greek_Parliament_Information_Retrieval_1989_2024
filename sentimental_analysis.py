import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import unicodedata
from db import get_db

# Function to remove accents from Greek text
def remove_accents(text):
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

# Establish database connection
engine = get_db()

# Load the data into a DataFrame
df = pd.read_sql('SELECT * FROM merged_speeches LIMIT 10000', engine)

# Load the lexicon with all polarity columns
lexicon = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')

# Preprocess the lexicon
lexicon = lexicon[['Term', 'Polarity1', 'Comments1']]

# Remove entries with ambiguous comments


# Drop rows with missing Term or Polarity1
lexicon = lexicon.dropna(subset=['Term', 'Polarity1'])

# Remove accentuation from terms
lexicon['Term'] = lexicon['Term'].apply(remove_accents)

# Define polarity to score mapping using Polarity1
def polarity_to_score(polarity):
    if polarity == 'POS':
        return 1
    elif polarity == 'NEG':
        return -1
    elif polarity == 'BOTH':
        return 0
    else:
        return 0

lexicon['score'] = lexicon['Polarity1'].apply(polarity_to_score)

# Create sentiment dictionary
sentiment_dict = dict(zip(lexicon['Term'], lexicon['score']))

# Load Greek language model
try:
    nlp = spacy.load('el_core_news_sm')
except OSError:
    print("SpaCy model 'el_core_news_sm' not found. Please install it using:")
    print("python -m spacy download el_core_news_sm")
    raise

# Function to preprocess text: tokenize, lemmatize, remove punctuation and spaces, and remove accents
def preprocess_text(text):
    try:
        doc = nlp(text)
        tokens = [
            remove_accents(token.lemma_.lower()) 
            for token in doc 
            if not token.is_punct and not token.is_space
        ]
        return tokens
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

# Function to calculate sentiment score
def calculate_sentiment(tokens, sentiment_dict):
    score = 0
    count = 0
    for token in tokens:
        if token in sentiment_dict:
            score += sentiment_dict[token]
            count += 1
    if count == 0:
        return 0
    return score / count

# Apply preprocessing and sentiment calculation
df['tokens'] = df['merged_speech'].apply(preprocess_text)
df['sentiment_score'] = df['tokens'].apply(lambda tokens: calculate_sentiment(tokens, sentiment_dict))

# Optional: Drop rows with missing 'political_party' or 'sentiment_score'
df = df.dropna(subset=['political_party'])

# Aggregate sentiment by political party
party_sentiment = df.groupby('political_party')['sentiment_score'].mean().reset_index()

# Sort parties by sentiment score for better visualization
party_sentiment = party_sentiment.sort_values(by='sentiment_score', ascending=False)

# Visualize the results
plt.figure(figsize=(14, 7))
sns.barplot(data=party_sentiment, x='political_party', y='sentiment_score', palette='viridis')
plt.title('Average Sentiment Score by Political Party', fontsize=16)
plt.ylabel('Average Sentiment Score', fontsize=14)
plt.xlabel('Political Party', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
