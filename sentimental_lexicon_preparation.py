import pandas as pd
import spacy
import pickle
from copy import deepcopy
from greek_stemmer import stemmer
import logging
from nltk.corpus import stopwords
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    # Load SpaCy model with disabled components to speed up processing
    nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])
except Exception as e:
    print("Please run this: python -m spacy download el_core_news_sm")
    exit(1)

# Define polarity mapping
POLARITY_MAPPING = {
    "POS": 1,
    "NEG": -1,
    "BOTH": 0,
    "N/A": 0  # Default to neutral for invalid or missing polarity
}

# Define unwanted patterns
UNWANTED_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)
TAB_PATTERN = re.compile(r'\t')


def prepWordSentiments(wordSentiments: pd.DataFrame) -> dict:
    """Preprocess the word sentiments lexicon to map words to their average polarity score."""
    def isNaN(subject) -> bool:
        """Check for NaN."""
        return subject != subject

    def cleanWord(word: str) -> str:
        """Clean and normalize words."""
        word = word.lower()
        if " " in word:
            components = word.split(" ")
            word = components[0]
            extraComponents = components[1].replace(" ", "").split("-")
            words = [word] + [word[:-2] + ending for ending in extraComponents[1:]]
            word = ",".join(words)
        return word

    def preprocessLookups(df: pd.DataFrame):
        df["Term"] = df["Term"].map(cleanWord)

    def computeAveragePolarity(lut: dict) -> dict:
        """Transform the lexicon into a dictionary mapping words to their average polarity score."""
        word_polarity = {}

        for item, values in lut.items():
            # Extract the four polarity columns
            polarity_values = values[8:12]
            # Map polarity strings to numerical values
            mapped_polarities = [POLARITY_MAPPING.get(val, 0) for val in polarity_values]
            # Compute average polarity
            average_polarity = sum(mapped_polarities) / len(mapped_polarities)
            # Handle NaN values by treating them as 0
            if any(isNaN(val) for val in mapped_polarities):
                average_polarity = 0
            word_polarity[item] = average_polarity

        return word_polarity

    preprocessLookups(wordSentiments)

    # Convert to dictionary for processing
    wordsDict = wordSentiments.set_index("Term").T.to_dict("list")
    secondDict = deepcopy(wordsDict)

    # Split multi-gender terms
    for word in list(wordsDict.keys()):
        if "," in word:
            words = word.split(",")
            for _word in words:
                secondDict[_word] = wordsDict[word]
            del secondDict[word]

    # Compute average polarity
    word_polarity = computeAveragePolarity(secondDict)
    return word_polarity


def stemWordSentiments(wordSentiments: dict, nlp) -> dict:
    """Stem word sentiments and map stemmed words to their average polarity score."""
    stemmedDict = {}
    failedList = []
    greek_stopwords = set(stopwords.words('greek'))

    for key, polarity in wordSentiments.items():
        try:
            # Process the word with SpaCy NLP pipeline
            doc = nlp(key)
            tokens = []

            for token in doc:
                # Remove unwanted patterns
                cleaned_token = UNWANTED_PATTERN.sub("", token.text)
                cleaned_token = TAB_PATTERN.sub("", cleaned_token)

                # Skip if token is empty, a stopword, or a single character
                if (
                    not cleaned_token
                    or cleaned_token.lower() in greek_stopwords
                    or len(cleaned_token) == 1
                ):
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

            # Combine stemmed tokens into a single string
            preprocessed_text = " ".join(tokens)

            if preprocessed_text:
                # If the stemmed word already exists, average the polarity
                if preprocessed_text in stemmedDict:
                    stemmedDict[preprocessed_text] = (stemmedDict[preprocessed_text] + polarity) / 2
                else:
                    stemmedDict[preprocessed_text] = polarity
            else:
                # If stemming results in an empty string, skip the word
                logging.warning(f"Stemming resulted in empty string for word '{key}'")
        except Exception as e:
            failedList.append((key, str(e)))
            continue

    # Log any failures
    if failedList:
        logging.warning(f"Failed to process {len(failedList)} words. Examples: {failedList[:5]}")

    return stemmedDict


if __name__ == "__main__":
    # Read the data from the TSV file
    logging.info("Reading word sentiments from file...")
    wordSentiments = pd.read_csv("lexicon/greek_sentiment_lexicon.tsv", sep="\t")

    # Preprocess the word sentiments
    logging.info("Preprocessing word sentiments...")
    word_polarity_dict = prepWordSentiments(wordSentiments)

    # Stem the word sentiments
    logging.info("Stemming word sentiments...")
    stemmedWordPolarityDict = stemWordSentiments(word_polarity_dict, nlp)

    # Save the stemmed word sentiments to a temporary CSV file
    logging.info("Saving stemmed word sentiments to a temporary CSV for verification...")
    temp_csv_path = "lexicon/temp_stemmed_word_polarity.csv"
    # Convert the dictionary to a DataFrame for easier inspection
    stemmed_df = pd.DataFrame(
        list(stemmedWordPolarityDict.items()),
        columns=["Term", "Average_Polarity"]
    )
    stemmed_df.to_csv(temp_csv_path, encoding="utf-8", index=False)
    logging.info(f"Stemmed word sentiments saved to {temp_csv_path} for inspection.")

    # Save the stemmed word sentiments to a pickle file
    with open("pkl_files/stemmedWordPolarity.pkl", "wb") as f:
        pickle.dump(stemmedWordPolarityDict, f)

    logging.info("Processing complete.")
