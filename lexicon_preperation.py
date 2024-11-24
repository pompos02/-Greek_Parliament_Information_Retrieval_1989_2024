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

# Load SpaCy model
nlp = spacy.load("el_core_news_sm")

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
    """Preprocess the word sentiments lexicon."""

    def isNaN(subject) -> bool:
        """Check for NaN."""
        return subject != subject

    def cleanWord(word: str) -> str:
        """Clean and normalize words."""
        word = word.lower()
        words = []
        if " " in word:
            components = word.split(" ")
            word = components[0]
            extraComponents = components[1].replace(" ", "").split("-")
            words = [word] + [word[:-2] + ending for ending in extraComponents[1:]]
            word = ",".join(words)
        return word

    def preprocessLookups(df: pd.DataFrame):
        df["Term"] = df["Term"].map(cleanWord)

    def reprocessLookups(lut: dict) -> dict:
        """Transform the lexicon into a structured dictionary."""
        for item in lut:
            newItem = {}
            lut[item] = lut[item][:-8]
            newItem["positions"] = lut[item][:4]
            newItem["subjectivity"] = lut[item][4:8]
            # Apply polarity mapping here
            newItem["polarity"] = [
                POLARITY_MAPPING.get(value, 0) for value in lut[item][8:12]
            ]
            newItem["anger"] = lut[item][12:16]
            newItem["disgust"] = lut[item][16:20]
            newItem["fear"] = lut[item][20:24]
            newItem["happiness"] = lut[item][24:28]
            newItem["sadness"] = lut[item][28:32]
            newItem["surprise"] = lut[item][32:36]

            badIndices = set()

            # Remove indices with NaN values
            for i in range(4):
                for key in newItem:
                    if isNaN(newItem[key][i]):
                        if key == "positions":
                            badIndices.add(i)
                        newItem[key][i] = 0

            # Combine duplicate positions
            foundPos = set()
            for i, pos in enumerate(newItem["positions"]):
                if pos in foundPos:
                    badIndices.add(i)
                foundPos.add(pos)

            # Drop invalid indices
            for key in newItem:
                newItem[key] = [newItem[key][i] for i in range(4) if i not in badIndices]

            lut[item] = newItem

    preprocessLookups(wordSentiments)

    # Convert to dictionary for processing
    wordsDict = wordSentiments.set_index("Term").T.to_dict("list")
    secondDict = deepcopy(wordsDict)

    # Split multi-gender terms
    for word in wordsDict:
        if "," in word:
            words = word.split(",")
            for _word in words:
                secondDict[_word] = wordsDict[word]
            del secondDict[word]

    reprocessLookups(secondDict)
    return secondDict


def stemWordSentiments(wordSentiments: dict, nlp) -> dict:
    """Stem word sentiments using the provided stemming logic."""
    secondDict = deepcopy(wordSentiments)
    failedList = []
    greek_stopwords = set(stopwords.words('greek'))
    for key in wordSentiments:
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
                    print(f"Error stemming word '{cleaned_token}': {e}")
                    continue

            # Combine stemmed tokens into a single string
            preprocessed_text = " ".join(tokens)

            # Update the dictionary with the stemmed key
            del secondDict[key]
            secondDict[preprocessed_text] = wordSentiments[key]
        except Exception as e:
            failedList.append((key, str(e)))
            continue

    # Log any failures
    if failedList:
        print(f"Failed to process {len(failedList)} words. Examples: {failedList[:5]}")

    return secondDict


if __name__ == "__main__":
    # Read the data from the TSV file
    logging.info("Reading word sentiments from file...")
    wordSentiments = pd.read_csv("greek_sentiment_lexicon.tsv", sep="\t")

    # Preprocess the word sentiments
    logging.info("Preprocessing word sentiments...")
    wordSentiments: dict = prepWordSentiments(wordSentiments)

    # Stem the word sentiments
    logging.info("Stemming word sentiments...")
    stemmedWordSentiments: dict = stemWordSentiments(wordSentiments, nlp)

    # Save the stemmed word sentiments to a temporary CSV file
    logging.info("Saving stemmed word sentiments to a temporary CSV for verification...")
    temp_csv_path = "temp_stemmed_word_sentiments.csv"
    # Convert the dictionary to a DataFrame for easier inspection
    stemmed_df = pd.DataFrame.from_dict(stemmedWordSentiments, orient="index")
    stemmed_df.to_csv(temp_csv_path, encoding="utf-8", index_label="Term")
    logging.info(f"Stemmed word sentiments saved to {temp_csv_path} for inspection.")

    # Save the stemmed word sentiments to a pickle file
    with open("pkl_files/stemmedWordSentiments.pickle", "wb") as f:
        pickle.dump(stemmedWordSentiments, f)

    logging.info("Processing complete.")
