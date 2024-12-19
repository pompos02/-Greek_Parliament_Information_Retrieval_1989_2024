import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import logging
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Ensure NLTK Greek stopwords are downloaded
nltk.download('stopwords')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load the preprocessed speeches from the merged pkl file
# This was computed by the preprocess_speeches_script.py
merged_pkl_path = 'pkl_files/preprocess_pkl_files/preprocessed_all_speeches.pkl'

logging.info('Loading preprocessed speeches...')
with open(merged_pkl_path, 'rb') as f:
    merged_df = pickle.load(f)
logging.info(f'Loaded {len(merged_df)} preprocessed speeches.')

# Define a mapping for the party names
party_mapping = {
    'νεα δημοκρατια': 'Νέα Δημοκρατία',
    'πανελληνιο σοσιαλιστικο κινημα': 'ΠΑΣΟΚ-ΚΙΝΗΜΑ ΑΛΛΑΓΗΣ',
    'δημοκρατικη συμπαραταξη (πανελληνιο σοσιαλιστικο κινημα - δημοκρατικη αριστερα)': 'ΠΑΣΟΚ-ΚΙΝΗΜΑ ΑΛΛΑΓΗΣ',
    'κινημα αλλαγης': 'ΠΑΣΟΚ-ΚΙΝΗΜΑ ΑΛΛΑΓΗΣ',
    'πασοκ-κινημα αλλαγης': 'ΠΑΣΟΚ-ΚΙΝΗΜΑ ΑΛΛΑΓΗΣ',
    'συνασπισμος ριζοσπαστικης αριστερας': 'Συνασπισμός Ριζοσπαστικής Αριστεράς',
    'συνασπισμος ριζοσπαστικης αριστερας - προοδευτικη συμμαχια': 'Συνασπισμός Ριζοσπαστικής Αριστεράς',
    'κομμουνιστικο κομμα ελλαδας': 'Κομμουνιστικό Κόμμα Ελλάδας',
    'λαικος συνδεσμος - χρυση αυγη': 'Λαϊκός Σύνδεσμος - Χρυσή Αυγή'
}


# Function to standardize party names
def standardize_party(party_name, mapping):
    if not isinstance(party_name, str):
        return 'Other'
    return mapping.get(party_name.lower(), 'Other')  # Assign 'Other' if not found


# Apply the mapping to create a new column 'standard_party'
merged_df['standard_party'] = merged_df['political_party'].apply(lambda x: standardize_party(x, party_mapping))

logging.info("Standard party distribution:")
party_distribution = merged_df['standard_party'].value_counts()
logging.info(f"\n{party_distribution}")

logging.info('Unique political_party values:')
unique_political_parties = merged_df['standard_party'].dropna().unique()
for party in unique_political_parties:
    logging.info(f"- {party}")

# Define the standardized party names to include in the analysis
selected_parties = [
    'Νέα Δημοκρατία',
    'ΠΑΣΟΚ-ΚΙΝΗΜΑ ΑΛΛΑΓΗΣ',
    'Συνασπισμός Ριζοσπαστικής Αριστεράς',
    'Κομμουνιστικό Κόμμα Ελλάδας',
    'Λαϊκός Σύνδεσμος - Χρυσή Αυγή'
]

# Filter the DataFrame to include only the selected parties
filtered_df = merged_df[merged_df['standard_party'].isin(selected_parties)].copy()
logging.info(f'Filtered speeches for selected parties. Remaining speeches: {len(filtered_df)}.')

# Load the processed lexicon from the pkl file
lexicon_pkl_path = 'pkl_files/stemmedWordPolarity.pkl'
logging.info('Loading sentiment lexicon...')
with open(lexicon_pkl_path, 'rb') as f:
    lexicon = pickle.load(f)
logging.info(f'Lexicon loaded with {len(lexicon)} entries.')


# Function to calculate polarity
def calculate_polarity(preprocessed_text, polarity_dict):
    tokens = preprocessed_text.split()
    score = 0
    count = 0
    for token in tokens:
        if token in polarity_dict:
            polarity = polarity_dict[token]
            score += polarity
            count += 1
    return score / count if count > 0 else 0


# Calculate polarity scores
filtered_df['balanced_polarity_score'] = filtered_df['preprocessed_speech'].apply(
    lambda text: calculate_polarity(text, lexicon)
)
logging.info('Balanced polarity scores calculated.')

# Ensure 'sitting_date' is in datetime format
filtered_df['sitting_date'] = pd.to_datetime(filtered_df['sitting_date'], errors='coerce')

# Extract the year from 'sitting_date'
filtered_df['year'] = filtered_df['sitting_date'].dt.year

# Drop rows with missing or invalid years
filtered_df = filtered_df.dropna(subset=['year'])
logging.info(f'Dropped rows with missing years. Remaining speeches: {len(filtered_df)}.')

# Group by standardized party and year, then calculate the mean balanced polarity score
polarity_trend = filtered_df.groupby(['standard_party', 'year'])['balanced_polarity_score'].mean().reset_index()

# Pivot the data for easier plotting
polarity_pivot = polarity_trend.pivot(index='year', columns='standard_party', values='balanced_polarity_score')

# Visualize the polarity trend over the years for each political party
plt.figure(figsize=(14, 8))
sns.lineplot(data=polarity_trend, x='year', y='balanced_polarity_score', hue='standard_party', marker='o')

plt.title('Balanced Polarity Trend Over the Years by Political Party', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Average Balanced Polarity Score', fontsize=14)
plt.legend(title='Political Party', fontsize=12, title_fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

top_n = 10  # Top 10 positive and negative words

# Separate lists to store top positive and top negative words data
top_positive_words_data = []
top_negative_words_data = []

# Retrieve Greek stopwords
greek_stopwords_set = set(stopwords.words('greek'))

for party in selected_parties:
    # Filter speeches for the current party
    party_df = filtered_df[filtered_df['standard_party'] == party]

    # Initialize Counters for positive and negative words
    word_counts_pos = Counter()
    word_counts_neg = Counter()

    # Iterate through each preprocessed speech
    for speech in party_df['preprocessed_speech']:
        words = speech.split()
        for word in words:
            if word in greek_stopwords_set:
                continue  # Skip stopwords
            polarity = lexicon.get(word, 0)
            if polarity > 0:
                word_counts_pos[word] += 1
            elif polarity < 0:
                word_counts_neg[word] += 1

    # Get top N positive words
    top_positive = word_counts_pos.most_common(top_n)
    # Get top N negative words
    top_negative = word_counts_neg.most_common(top_n)

    # Append top positive words to the positive list
    for word, count in top_positive:
        top_positive_words_data.append({
            'party': party,
            'word': word,
            'frequency': count
        })

    # Append top negative words to the negative list
    for word, count in top_negative:
        top_negative_words_data.append({
            'party': party,
            'word': word,
            'frequency': count
        })

# Create DataFrames for positive and negative top words
top_positive_words_df = pd.DataFrame(top_positive_words_data)
top_negative_words_df = pd.DataFrame(top_negative_words_data)

# Add a 'polarity' column to each DataFrame
top_positive_words_df['polarity'] = 'positive'
top_negative_words_df['polarity'] = 'negative'

# Save the positive top words to a CSV file
output_csv_path_top_positive = 'app/static/data/top_positive_words_by_party.csv'
top_positive_words_df.to_csv(output_csv_path_top_positive, index=False, encoding='utf-8-sig')
logging.info(f'Top positive words by party exported to {output_csv_path_top_positive}.')

# Save the negative top words to a CSV file
output_csv_path_top_negative = 'app/static/data/top_negative_words_by_party.csv'
top_negative_words_df.to_csv(output_csv_path_top_negative, index=False, encoding='utf-8-sig')
logging.info(f'Top negative words by party exported to {output_csv_path_top_negative}.')
