import pickle
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, select, Table, MetaData,asc
import matplotlib.pyplot as plt
from db import get_db
import os

# Load TF-IDF Matrix and Feature Names
def load_tfidf_data():
    # Load the TF-IDF matrix
    with open('pkl_files/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    
    # Load the vectorizer to get feature names
    with open('pkl_files/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names



def get_top_keywords(tfidf_matrix, feature_names, top_n=10):
    COMMON_KEYWORDS = {
    "αυτ", "εινα", "της", "αυτ", "απο", "εν", "τους", "κυρι"
    , "οτ", "δε", "απο","αλλ","εχ","λογ","μι","οποι"
    # Add more common keywords you want to exclude
    }

    keywords = []
    for i in range(tfidf_matrix.shape[0]):  # Iterate over rows
        row = tfidf_matrix[i].toarray().flatten()  # Convert sparse row to dense
        indices = np.argsort(row)[::-1]  # Sort indices by TF-IDF score
        filtered_keywords = [
            (feature_names[idx], row[idx])
            for idx in indices if feature_names[idx] not in COMMON_KEYWORDS
        ][:top_n]  # Only take top N after filtering
        keywords.append(filtered_keywords)
    return keywords



# Connect to PostgreSQL Database
from sqlalchemy import MetaData, Table

def connect_to_db():
    engine = get_db()  # Call the `get_db` function to get the engine
    metadata = MetaData()
    merged_speeches = Table("merged_speeches", metadata, autoload_with=engine)  # Use the engine for autoload
    return engine, merged_speeches

# Fetch Speech Data
def fetch_speech_data(engine, merged_speeches):
    query = select(
        merged_speeches.c.id,
        merged_speeches.c.member_name,
        merged_speeches.c.political_party,
        merged_speeches.c.sitting_date,
    ).order_by(asc(merged_speeches.c.sitting_date))  # Add ORDER BY sitting_date ASC

    with engine.connect() as conn:
        result = conn.execute(query)
        return result.fetchall()

# Analyze Keywords by Member and Party
def analyze_keywords(speech_data, extracted_keywords):
    # Convert data into a Pandas DataFrame
    df = pd.DataFrame(
        speech_data, 
        columns=["id", "member_name", "political_party", "sitting_date"]
    )
    df["keywords"] = extracted_keywords  # Add keywords column

    # Group by member to find top keywords with their scores
    def aggregate_keywords(keywords_list):
        # Flatten the list of keywords for all speeches
        flattened = [item for sublist in keywords_list for item in sublist]
        # Aggregate scores for each keyword
        keyword_scores = {}
        for keyword, score in flattened:
            keyword_scores[keyword] = keyword_scores.get(keyword, 0) + score
        # Sort by score and return top 10
        return sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    member_keywords = df.groupby("member_name")["keywords"].apply(aggregate_keywords)
    party_keywords = df.groupby("political_party")["keywords"].apply(aggregate_keywords)

    return df, member_keywords, party_keywords




# Track Keyword Changes Over Time
def track_member_keyword_trends(df):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords for time-based analysis
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract word
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Group by member, year, and keyword to sum scores
    member_keyword_trends = exploded.groupby(["member_name", "sitting_year", "keyword"])["score"].sum().reset_index()

    return member_keyword_trends

def track_party_keyword_trends(df):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords for time-based analysis
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract word
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Group by party, year, and keyword to sum scores
    party_keyword_trends = exploded.groupby(["political_party", "sitting_year", "keyword"])["score"].sum().reset_index()

    return party_keyword_trends

# Visualize Keyword Trends
def visualize_keyword_trends_trends(df_trends, group_by, title, plot_file):
    """
    Visualize keyword trends over time for a given group (member or party).

    Parameters:
        df_trends (pd.DataFrame): DataFrame containing trends with columns [group, year, keyword, score]
        group_by (str): The column to group by ('member_name' or 'political_party')
        title (str): The title of the plot
        plot_file (str): The file path to save the plot
    """
    # Determine top 10 keywords based on total scores across all groups and years
    top_keywords = df_trends.groupby("keyword")["score"].sum().sort_values(ascending=False).head(10).index.tolist()

    # Filter the trends to include only top keywords
    df_top = df_trends[df_trends["keyword"].isin(top_keywords)]

    # Pivot the DataFrame to have years as x-axis and keywords as columns
    pivot_df = df_top.pivot_table(index="sitting_year", columns="keyword", values="score", aggfunc='sum', fill_value=0)

    # Plotting
    plt.figure(figsize=(16, 12))
    pivot_df.plot(kind="line", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Aggregated TF-IDF Score", fontsize=14)
    plt.legend(title="Keywords", fontsize=6, title_fontsize=6)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")

def visualize_keyword_trends(df_trends, top_keywords=None):
    if top_keywords:
        df_trends = df_trends[top_keywords]

    df_trends.plot(kind="line", figsize=(16, 12))
    plt.title("Keyword Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Keyword Frequency")
    plt.legend(title="Keywords")
    plt.tight_layout()
    plt.show()

def load_speech_ids():
    with open('pkl_files/speech_ids.pkl', 'rb') as f:
        speech_ids = pickle.load(f)
    return speech_ids

# Main Execution Flow
def main():


    # Load TF-IDF matrix, feature names, and speech IDs
    tfidf_matrix, feature_names = load_tfidf_data()
    speech_ids = load_speech_ids()

    # Connect to the database
    engine, merged_speeches = connect_to_db()

    # Fetch speech data
    speech_data = fetch_speech_data(engine, merged_speeches)

    # Convert speech data to a DataFrame
    speech_data_df = pd.DataFrame(
        speech_data,
        columns=["id", "member_name", "political_party", "sitting_date"]
    )

    # Align the speech data with the speech IDs
    speech_data_df = speech_data_df[speech_data_df["id"].isin(speech_ids)]

    # Sort speech_data_df by `id` to ensure it matches the TF-IDF matrix
    speech_data_df = speech_data_df.set_index("id").loc[speech_ids].reset_index()

    # Extract top keywords
    extracted_keywords = get_top_keywords(tfidf_matrix, feature_names)

    # Analyze keywords by member and party
    df, member_keywords, party_keywords = analyze_keywords(
        speech_data_df.values, extracted_keywords
    )

    # 1. Track keyword trends over time for members
    member_keyword_trends = track_member_keyword_trends(df)

    # 2. Track keyword trends over time for parties
    party_keyword_trends = track_party_keyword_trends(df)

    # 3. Visualize Member Keyword Trends
    member_plot_file = "output/member_keyword_trends_over_time.png"
    visualize_keyword_trends_trends(
        df_trends=member_keyword_trends,
        group_by="member_name",
        title="Top 10 Member Keywords Trends Over Time",
        plot_file=member_plot_file
    )

    # 4. Visualize Party Keyword Trends
    party_plot_file = "output/party_keyword_trends_over_time.png"
    visualize_keyword_trends_trends(
        df_trends=party_keyword_trends,
        group_by="political_party",
        title="Top 10 Party Keywords Trends Over Time",
        plot_file=party_plot_file
    )

    # **Optional: Save Keyword Trends to CSV for Further Analysis**
    keyword_trends_file = "output/keyword_trends_over_time.csv"
    # Combine member and party trends into a single CSV if needed
    member_keyword_trends.to_csv("output/member_keyword_trends_over_time.csv", index=False)
    party_keyword_trends.to_csv("output/party_keyword_trends_over_time.csv", index=False)
    print("Keyword trends saved to CSV files.")




if __name__ == "__main__":
    main()

