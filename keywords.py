import pickle
import numpy as np
import pandas as pd
from sqlalchemy import select, Table, MetaData
import matplotlib.pyplot as plt
from db import get_db


def load_tfidf_data():
    # Load the TF-IDF matrix
    with open('pkl_files/tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)

    # Load the vectorizer to get feature names
    with open('pkl_files/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names


# Load Speech IDs
def load_speech_ids():
    with open('pkl_files/speech_ids.pkl', 'rb') as f:
        speech_ids = pickle.load(f)
    return speech_ids


def get_top_keywords(tfidf_matrix, feature_names, top_n=10):
    COMMON_KEYWORDS = {
        "αυτ", "εινα", "της", "αυτ", "απο", "εν", "τους", "κυρι",
        "οτ", "δε", "απο", "αλλ", "εχ", "λογ", "μι"
    }

    keywords = []
    for i in range(tfidf_matrix.shape[0]):  # Iterate over rows
        row = tfidf_matrix[i].toarray().flatten()  # Convert sparse row to dense
        indices = np.argsort(row)[::-1]  # Sort indices by TF-IDF score descending
        filtered_keywords = [
                                (feature_names[idx], row[idx])
                                for idx in indices if feature_names[idx] not in COMMON_KEYWORDS
                            ][:top_n]  # Only take top N after filtering
        keywords.append(filtered_keywords)
    return keywords


# Connect to the PostgreSQL Database
def connect_to_db():
    engine = get_db()
    metadata = MetaData()
    merged_speeches = Table("merged_speeches", metadata, autoload_with=engine)
    return engine, merged_speeches


# Fetch Speech Data using Speech IDs
def fetch_speech_data(engine, merged_speeches, speech_ids):
    # Build the query to select speeches matching the speech_ids
    query = select(
        merged_speeches.c.id,
        merged_speeches.c.member_name,
        merged_speeches.c.political_party,
        merged_speeches.c.sitting_date,
    ).where(
        merged_speeches.c.id.in_(speech_ids)
    )

    with engine.connect() as conn:
        result = conn.execute(query)
        speech_data = result.fetchall()

    # Convert to DataFrame for easier handling
    speech_data_df = pd.DataFrame(
        speech_data,
        columns=["id", "member_name", "political_party", "sitting_date"]
    )

    # Create a DataFrame of speech_ids with their order
    speech_ids_df = pd.DataFrame({'id': speech_ids, 'tfidf_row_index': range(len(speech_ids))})

    # Merge speech_data_df with speech_ids_df on 'id'
    merged_df = pd.merge(speech_ids_df, speech_data_df, on='id', how='left')

    # Check for any missing data
    missing_data = merged_df[merged_df['member_name'].isnull()]
    if not missing_data.empty:
        print("Warning: The following speech IDs are missing from the database:")
        print(missing_data['id'].tolist())
        merged_df = merged_df.dropna(subset=['member_name'])

    # Sort merged_df by 'tfidf_row_index' to ensure alignment with TF-IDF matrix
    merged_df = merged_df.sort_values('tfidf_row_index').reset_index(drop=True)

    return merged_df


# Analyze Keywords by Member and Party
def analyze_keywords(df):
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

    return member_keywords, party_keywords


# Track Keyword Changes Over Time for Members
def track_member_keyword_trends(df, member_keywords):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords for time-based analysis
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract word
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Merge with top keywords to filter
    exploded = exploded.merge(
        member_keywords.apply(lambda x: [k for k, _ in x]).rename("top_keywords"),
        left_on="member_name",
        right_index=True
    )

    # Keep only rows where keyword is in top_keywords
    exploded = exploded[exploded.apply(lambda row: row["keyword"] in row["top_keywords"], axis=1)]

    # Group by member, year, and keyword to sum scores
    member_keyword_trends = exploded.groupby(["member_name", "sitting_year", "keyword"])["score"].sum().reset_index()

    return member_keyword_trends


# Track Keyword Changes Over Time for Parties
def track_party_keyword_trends(df, party_keywords):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords for time-based analysis
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract word
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Merge with top keywords to filter
    exploded = exploded.merge(
        party_keywords.apply(lambda x: [k for k, _ in x]).rename("top_keywords"),
        left_on="political_party",
        right_index=True
    )

    # Keep only rows where keyword is in top_keywords
    exploded = exploded[exploded.apply(lambda row: row["keyword"] in row["top_keywords"], axis=1)]

    # Group by party, year, and keyword to sum scores
    party_keyword_trends = exploded.groupby(["political_party", "sitting_year", "keyword"])["score"].sum().reset_index()

    return party_keyword_trends


# Visualize Keyword Trends
def visualize_keyword_trends_trends(df_trends, group_by, title, plot_file):
    """ Visualize keyword trends over time for a given group (member or party)."""
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


def compute_top_keywords_per_year(df, top_n=10):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords to have one keyword per row
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract keyword
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Group by year and keyword to sum scores
    yearly_keyword_scores = exploded.groupby(["sitting_year", "keyword"])["score"].sum().reset_index()

    # For each year, get top N keywords
    top_keywords_per_year = yearly_keyword_scores.groupby("sitting_year").apply(
        lambda x: x.nlargest(top_n, 'score')
    ).reset_index(drop=True)

    return top_keywords_per_year


def compute_top_keywords_overall_and_trend(df, top_n=10):
    df["sitting_year"] = pd.to_datetime(df["sitting_date"]).dt.year

    # Explode keywords to have one keyword per row
    exploded = df.explode("keywords")
    exploded["keyword"] = exploded["keywords"].apply(lambda x: x[0])  # Extract keyword
    exploded["score"] = exploded["keywords"].apply(lambda x: x[1])  # Extract score

    # Aggregate total scores for all keywords
    total_keyword_scores = exploded.groupby("keyword")["score"].sum().reset_index()

    # Get top N keywords overall
    top_keywords_overall = total_keyword_scores.nlargest(top_n, 'score')['keyword'].tolist()

    # Filter exploded DataFrame to include only top keywords
    exploded_top_keywords = exploded[exploded["keyword"].isin(top_keywords_overall)]

    # Group by year and keyword to sum scores
    keyword_trends = exploded_top_keywords.groupby(["sitting_year", "keyword"])["score"].sum().reset_index()

    return top_keywords_overall, keyword_trends


def visualize_overall_keyword_trends(keyword_trends, top_keywords_overall, title, plot_file):
    # Pivot the DataFrame to have years as x-axis and keywords as columns
    pivot_df = keyword_trends.pivot_table(index="sitting_year", columns="keyword", values="score", aggfunc='sum',
                                          fill_value=0)

    # Ensure the columns are ordered according to top_keywords_overall
    pivot_df = pivot_df[top_keywords_overall]

    # Plotting
    plt.figure(figsize=(16, 12))
    pivot_df.plot(kind="line", linewidth=1.5)
    plt.title(title, fontsize=16)
    plt.xlabel("Year", fontsize=14)
    plt.ylabel("Aggregated TF-IDF Score", fontsize=14)
    plt.legend(title="Keywords", fontsize=8, title_fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")


def main():
    # Load TF-IDF matrix, feature names, and speech IDs
    tfidf_matrix, feature_names = load_tfidf_data()
    speech_ids = load_speech_ids()

    # Connect to the database
    engine, merged_speeches = connect_to_db()

    # Fetch speech data using speech IDs
    df = fetch_speech_data(engine, merged_speeches, speech_ids)

    # Ensure the number of speeches matches
    if tfidf_matrix.shape[0] != len(df):
        print("Warning: The number of speeches in the TF-IDF matrix does not match the speech data.")
        print(f"TF-IDF matrix rows: {tfidf_matrix.shape[0]}, Speech data rows: {len(df)}")
        # Adjust tfidf_matrix to match the speeches available in df
        tfidf_indices = df['tfidf_row_index'].tolist()
        tfidf_matrix = tfidf_matrix[tfidf_indices]

    # Extract top keywords
    extracted_keywords = get_top_keywords(tfidf_matrix, feature_names)

    # Add keywords to the DataFrame
    df["keywords"] = extracted_keywords

    # Analyze keywords by member and party
    member_keywords, party_keywords = analyze_keywords(df)

    print("Number of keywords per member:")
    print(member_keywords.apply(len).head())

    print("Number of keywords per party:")
    print(party_keywords.apply(len).head())

    # Track keyword trends over time for members
    member_keyword_trends = track_member_keyword_trends(df, member_keywords)

    # Track keyword trends over time for parties
    party_keyword_trends = track_party_keyword_trends(df, party_keywords)

    # Visualize Member Keyword Trends
    member_plot_file = "output/member_keyword_trends_over_time.png"
    visualize_keyword_trends_trends(
        df_trends=member_keyword_trends,
        group_by="member_name",
        title="Top 10 Member Keywords Trends Over Time",
        plot_file=member_plot_file
    )

    # Visualize Party Keyword Trends
    party_plot_file = "output/party_keyword_trends_over_time.png"
    visualize_keyword_trends_trends(
        df_trends=party_keyword_trends,
        group_by="political_party",
        title="Top 10 Party Keywords Trends Over Time",
        plot_file=party_plot_file
    )

    # Compute top keywords per year and output to CSV
    top_keywords_per_year = compute_top_keywords_per_year(df, top_n=10)
    top_keywords_per_year.to_csv("output/top_keywords_per_year.csv", index=False)
    print("Top keywords per year saved to output/top_keywords_per_year.csv")

    # Compute top keywords overall and track their trends
    top_keywords_overall, keyword_trends = compute_top_keywords_overall_and_trend(df, top_n=10)

    # Save top keywords overall to CSV
    pd.DataFrame({'keyword': top_keywords_overall}).to_csv("output/top_keywords_overall.csv", index=False)
    print("Top keywords overall saved to output/top_keywords_overall.csv")

    # Save keyword trends to CSV
    keyword_trends.to_csv("output/keyword_trends_overall.csv", index=False)
    print("Keyword trends over time saved to output/keyword_trends_overall.csv")

    # Visualize top keywords overall trends
    overall_plot_file = "output/top_keywords_overall_trends.png"
    visualize_overall_keyword_trends(
        keyword_trends=keyword_trends,
        top_keywords_overall=top_keywords_overall,
        title="Top 10 Keywords Overall Trends Over Time",
        plot_file=overall_plot_file
    )

    # Save member and party keyword trends to CSV
    member_keyword_trends.to_csv("output/member_keyword_trends_over_time.csv", index=False)
    party_keyword_trends.to_csv("output/party_keyword_trends_over_time.csv", index=False)
    print("Keyword trends saved to CSV files.")


if __name__ == "__main__":
    main()
