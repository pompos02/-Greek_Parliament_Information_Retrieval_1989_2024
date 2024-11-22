import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForPreTraining, AutoModel, TextClassificationPipeline
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from db import get_db

# Set up the device for GPU usage if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Step 1: Load Your Data
# Replace with your actual database connection string and table name
engine = get_db()

# Load the data into a DataFrame
df = pd.read_sql('SELECT * FROM merged_speeches', engine)



# Load the tokenizer and model
model_name = "pchatz/palobert-base-greek-social-media-sentiment-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print(model.config)

device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
sentiment_pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device, return_all_scores=True)

# Define a function to get sentiment scores
def get_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])  # Truncate text to 512 tokens if necessary
        # The model outputs scores for each class; adjust according to your model's classes
        # Assuming the classes are ['negative', 'neutral', 'positive']
        scores = result[0]
        scores_dict = {item['label']: item['score'] for item in scores}
        return scores_dict
    except Exception as e:
        print(f"Error processing text: {e}")
        return None

# Apply the function to your DataFrame
tqdm.pandas()
df['sentiment_scores'] = df['merged_speech'].progress_apply(get_sentiment)


# Specify the political party you're interested in
target_party = 'ελληνικη λυση - κυριακος βελοπουλος'  # Replace with the actual party name

# Filter the DataFrame
df_party = df[df['political_party'] == target_party].copy()

# Ensure there are sentiment scores
df_party = df_party[df_party['sentiment_scores'].notnull()]

# Assuming the sentiment labels are 'negative', 'neutral', 'positive'
def extract_sentiment_label(scores_dict):
    if scores_dict is None:
        return None
    return max(scores_dict, key=scores_dict.get)

def extract_sentiment_score(scores_dict):
    if scores_dict is None:
        return None
    return max(scores_dict.values())

df_party['sentiment_label'] = df_party['sentiment_scores'].apply(extract_sentiment_label)
df_party['sentiment_score'] = df_party['sentiment_scores'].apply(extract_sentiment_score)

import matplotlib.pyplot as plt
import seaborn as sns

# Count of sentiment labels
plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment_label', data=df_party)
plt.title(f'Sentiment Distribution for {target_party}')
plt.xlabel('Sentiment')
plt.ylabel('Number of Speeches')
plt.show()
