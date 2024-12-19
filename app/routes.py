from flask import render_template, request
import pandas as pd
from sqlalchemy import text
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
import spacy
from greek_stemmer import stemmer
from db import get_db
from task3_top_k_pairs import load_pickle_files, get_top_k_similar_pairs, get_political_parties

engine = get_db()


def create_routes(app):
    # Load the TF-IDF components at startup
    def load_tfidf_components():
        try:
            with open('pkl_files/tfidf_vectorizer.pkl', 'rb') as f:
                vectorizer = pickle.load(f)

            with open('pkl_files/tfidf_matrix.pkl', 'rb') as f:
                tfidf_matrix = pickle.load(f)

            with open('pkl_files/speech_ids.pkl', 'rb') as f:
                speech_ids = pickle.load(f)

            return vectorizer, tfidf_matrix, speech_ids
        except Exception as e:
            print(f"Error loading TF-IDF components: {e}")
            return None, None, None

    # Initialize components
    vectorizer, tfidf_matrix, speech_ids = load_tfidf_components()

    # Initialize SpaCy
    try:
        nlp = spacy.load("el_core_news_sm", disable=["parser", "ner"])
    except Exception as e:
        print(f"Error loading SpaCy model: {e}")
        exit(1)

    # Initialize other components
    greek_stopwords = set(stopwords.words('greek'))
    UNWANTED_PATTERN = re.compile(r'[0-9@#$%^&*()\-\_=+\[\]{};:\'",.<>/?\\|`~!]')
    TAB_PATTERN = re.compile(r'\t+')

    def preprocess_query(query_text):
        """Preprocess the search query using the same pipeline as the documents."""
        # Process with SpaCy
        doc = nlp(query_text)
        tokens = []

        for token in doc:
            cleaned_token = UNWANTED_PATTERN.sub('', token.text)
            cleaned_token = TAB_PATTERN.sub('', cleaned_token)

            if (not cleaned_token) or (cleaned_token.lower() in greek_stopwords) or (len(cleaned_token) == 1):
                continue

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

        return ' '.join(tokens)

    def get_speech_excerpt(speech, max_length=20):
        """Create a relevant excerpt from the speech text"""
        if len(speech) <= max_length:
            return speech
        return speech[:max_length].strip() + "..."

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/search_engine')
    def search():
        query = request.args.get('q', '')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)

        if not query:
            return render_template('search.html')

        try:
            # Check if components are loaded
            if None in (vectorizer, tfidf_matrix, speech_ids):
                return render_template('search.html',
                                       error='Search system not properly initialized')

            # Preprocess the query
            processed_query = preprocess_query(query)

            # Transform query to TF-IDF vector
            query_vector = vectorizer.transform([processed_query])

            similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
            print(similarity_scores.shape)
            # Get top matches above threshold
            threshold = 0.1
            top_indices = similarity_scores.argsort()[::-1]

            # Calculate pagination
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            # Filter results above threshold
            results = []
            for idx in top_indices:
                if similarity_scores[idx] < threshold:
                    break
                results.append({
                    'speech_id': int(speech_ids[idx]),
                    'similarity_score': float(similarity_scores[idx])
                })

            total_results = len(results)
            paginated_results = results[start_idx:end_idx]

            if not paginated_results:
                return render_template('search.html',
                                       query=query,
                                       no_results=True)

            # Get speech IDs for SQL query
            speech_ids_to_fetch = [r['speech_id'] for r in paginated_results]

            # Fetch speeches from database
            sql_query = text("""
                SELECT id, merged_speech, member_name, sitting_date , political_party,parliamentary_period,parliamentary_session,parliamentary_sitting
                FROM merged_speeches 
                WHERE id = ANY(:speech_ids)
            """)

            with engine.connect() as connection:
                result = connection.execute(sql_query, {'speech_ids': speech_ids_to_fetch})
                speeches = result.mappings().all()

            # Convert to dictionary for easier lookup
            speeches_dict = {speech['id']: speech for speech in speeches}

            # Combine database results with similarity scores
            final_results = []
            for result in paginated_results:
                speech = speeches_dict.get(result['speech_id'])
                if speech:
                    final_results.append({
                        'id': int(speech['id']),
                        'merged_speech': get_speech_excerpt(speech['merged_speech']),
                        'member_name': speech['member_name'] or 'Unknown',
                        'political_party': speech['political_party'],
                        'score': result['similarity_score'],
                        'parliamentary_period': speech['parliamentary_period'],
                        'parliamentary_session': speech['parliamentary_session'],
                        'parliamentary_sitting': speech['parliamentary_sitting'],
                        'sitting_date': speech['sitting_date'] if speech['sitting_date'] else None,

                    })

            # Calculate pagination info
            total_pages = (total_results + per_page - 1) // per_page
            has_next = page < total_pages
            has_prev = page > 1

            print(f"Query: {query}")
            print(f"Processed Query: {processed_query}")

            return render_template('search.html',
                                   query=query,
                                   results=final_results,
                                   page=page,
                                   total_pages=total_pages,
                                   has_next=has_next,
                                   has_prev=has_prev,
                                   total_results=total_results)

        except Exception as e:
            print(f"Search error: {e}")
            return render_template('search.html',
                                   query=query,
                                   error='An error occurred while processing your search')

    @app.route('/merged_speech')
    def view_merged_speech():
        # Get parameters from the URL
        period = request.args.get('period')
        session = request.args.get('session')
        sitting = request.args.get('sitting')
        date = request.args.get('date')
        member_name = request.args.get('member_name')
        query = request.args.get('query', '')

        try:
            # Fetch the merged speech from the merged_speeches table
            sql_query = text("""
                SELECT member_name, parliamentary_period, parliamentary_session, parliamentary_sitting, sitting_date, merged_speech
                FROM merged_speeches
                WHERE parliamentary_period = :period
                AND parliamentary_session = :session
                AND parliamentary_sitting = :sitting
                AND sitting_date = :date
                AND member_name LIKE :member_name
            """)

            with engine.connect() as connection:
                result = connection.execute(sql_query,
                                            {'period': period, 'session': session, 'sitting': sitting, 'date': date,
                                             'member_name': member_name})
                merged_speech = result.fetchone()

            if merged_speech:
                return render_template('merged_speech.html', speech=merged_speech, query=query)
            else:
                return render_template('merged_speech.html',
                                       error="No merged speech found for the provided parameters.", query=query)
        except Exception as e:
            print(f"Error fetching merged speech: {e}")
            return render_template('merged_speech.html', error="An error occurred while retrieving the merged speech.",
                                   query=query)

    @app.route('/keywords')
    def keywords():
        return render_template('keywords.html')

    @app.route('/top_k_member_pairs', methods=['GET', 'POST'])
    def top_k_member_pairs():
        if request.method == 'POST':
            try:
                k = int(request.form.get('k', 10))
                tfidf_matrix, speech_ids = load_pickle_files()

                # Convert speech_ids into a list of member names
                member_names = [speech_id[1] if isinstance(speech_id, tuple) else speech_id for speech_id in speech_ids]

                cosine_sim_matrix = cosine_similarity(tfidf_matrix)

                # Top-k most similar pairs
                top_k_pairs = get_top_k_similar_pairs(cosine_sim_matrix, member_names, k)

                # Results with member names, political parties, and similarities
                results = []
                for i, ((member1, member2), similarity) in enumerate(top_k_pairs.items(), start=1):
                    member1_parties = get_political_parties(member1)
                    member2_parties = get_political_parties(member2)
                    results.append({
                        "rank": i,
                        "member1": member1,
                        "member1_parties": member1_parties,
                        "member2": member2,
                        "member2_parties": member2_parties,
                        "similarity": round(similarity, 4),
                    })

                return render_template('top_k_member_pairs.html', results=results)

            except Exception as e:
                return render_template('top_k_member_pairs.html', error=f"An error occurred: {str(e)}")

        return render_template('top_k_member_pairs.html')

    @app.route('/top_concepts')
    def top_concepts():
        return render_template('top_concepts.html')

    @app.route('/top_concepts/<concept_type>')
    def view_concept_file(concept_type):
        if concept_type == 'lsi':
            filename = 'txt/lsi_concepts.txt'
            title = "LSI Results"
        elif concept_type == 'nmf':
            filename = 'txt/nmf_concepts.txt'
            title = "NMF Results"
        else:
            return "Invalid concept type.", 404

        try:
            file_path = f"app/static/{filename}"
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.readlines()

            return render_template('concepts_display.html', title=title, content=content)
        except Exception as e:
            return f"Error retrieving the file: {e}", 500

    @app.route('/speeches_clustering')
    def speeches_clustering():
        return render_template('speeches_clustering.html')

    @app.route('/clustering/<method>')
    def view_clustering(method):
        if method == 'kmeans_lsi':
            filename = 'txt/best_speeches_per_cluster_lsi.txt'
            title = "K-Means Clustering (LSI)"
        elif method == 'kmeans_nmf':
            filename = 'txt/best_speeches_per_cluster_nmf.txt'
            title = "K-Means Clustering (NMF)"
        else:
            return "Invalid clustering method.", 404

        try:

            file_path = f"app/static/{filename}"
            formatted_content = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if "Speech ID:" not in line:  # Skip lines containing "Speech ID"
                        formatted_content.append(line.strip())

            return render_template('clustering_display.html', title=title, content=formatted_content)
        except Exception as e:
            return f"Error retrieving the file: {e}", 500

    @app.route('/polarity_tracking')
    def polarity_tracking():
        return render_template('polarity_tracking.html')

    @app.route('/polarity_data/<data_type>')
    def polarity_data(data_type):
        if data_type == "positive":
            csv_path = 'app/static/data/top_positive_words_by_party.csv'
            title = "Top Positive Words by Party"
        elif data_type == "negative":
            csv_path = 'app/static/data/top_negative_words_by_party.csv'
            title = "Top Negative Words by Party"
        else:
            return "Invalid data type.", 404

        try:
            data = pd.read_csv(csv_path)
            return render_template(
                'polarity_data.html',
                title=title,
                columns=data.columns.tolist(),
                rows=data.values.tolist()
            )
        except Exception as e:
            return f"Error reading the CSV file: {e}", 500

