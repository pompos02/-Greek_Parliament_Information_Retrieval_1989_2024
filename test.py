import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import unicodedata
from db import get_db
from collections import Counter
import logging

def remove_accents(text):
    if not isinstance(text, str):
        return ''
    return ''.join(
        char for char in unicodedata.normalize('NFD', text)
        if unicodedata.category(char) != 'Mn'
    )

# Function to determine if a comment is not highly ambiguous
def is_not_highly_ambiguous(comment):
    if isinstance(comment, str):
        return 'highly ambiguous' not in comment.lower()
    return True  # If not a string, consider it as not highly ambiguous

# Define polarity to score mapping
def polarity_to_score(polarity):
    if polarity == 'POS':
        return 1
    elif polarity == 'NEG':
        return -1
    elif polarity == 'BOTH':
        return 0
    else:
        return np.nan  # Use NaN for 'N/A' or undefined polarities


lexicon = pd.read_csv('greek_sentiment_lexicon.tsv', sep='\t')
logging.info(f'Lexicon loaded with {len(lexicon)} records.')

# Preprocess the lexicon: select relevant columns
# Include all Polarity columns (assuming there are up to Polarity4)
polarity_columns = ['Polarity1', 'Polarity2', 'Polarity3', 'Polarity4']
lexicon = lexicon[['Term'] + polarity_columns + ['Comments1']]
logging.info('Selected relevant columns from lexicon.')

# Apply the filter to exclude highly ambiguous terms
lexicon = lexicon[
    lexicon['Comments1'].isna() | lexicon['Comments1'].apply(is_not_highly_ambiguous)
]
logging.info(f'Lexicon filtered for ambiguity. Remaining terms: {len(lexicon)}.')

# Drop rows with missing Term
lexicon = lexicon.dropna(subset=['Term'])
logging.info(f'Lexicon after dropping rows with missing Term: {len(lexicon)}.')

# Remove accentuation from terms
lexicon['Term'] = lexicon['Term'].apply(remove_accents)
logging.info('Accentuation removed from lexicon terms.')

# Apply polarity_to_score to all polarity columns
for col in polarity_columns:
    lexicon[col] = lexicon[col].apply(polarity_to_score)

# Calculate the average sentiment score across available polarity columns
lexicon['average_score'] = lexicon[polarity_columns].mean(axis=1, skipna=True)

# Alternatively, you can use majority voting or other aggregation methods
# For simplicity, we'll use the average here

# Drop rows where average_score is NaN (i.e., all polarity columns were 'N/A' or undefined)
lexicon = lexicon.dropna(subset=['average_score'])
logging.info(f'Lexicon after calculating average sentiment scores: {len(lexicon)}.')

# Create sentiment dictionary
sentiment_dict = dict(zip(lexicon['Term'], lexicon['average_score']))
logging.info(f'Sentiment dictionary created with {len(sentiment_dict)} terms.')

# Analyze sentiment dictionary balance
sentiment_counts = Counter(sentiment_dict.values())
logging.info(f'Sentiment dictionary counts: {sentiment_counts}')
# Assuming sentiment_dict is already created
sentiment_counts = Counter(sentiment_dict.values())

# Convert to DataFrame for easier plotting
sentiment_df = pd.DataFrame(list(sentiment_counts.items()), columns=['Sentiment_Score', 'Count'])

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(data=sentiment_df, x='Sentiment_Score', y='Count', palette='Reds')
plt.title('Distribution of Sentiment Scores in Sentiment Dictionary')
plt.xlabel('Sentiment Score')
plt.ylabel('Number of Terms')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
