from flask import render_template, request, jsonify

from sqlalchemy import text, create_engine
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
import re
from nltk.corpus import stopwords
import spacy
from greek_stemmer import stemmer

db_name = 'speeches'
db_user = 'pompos02'            # Replace with your PostgreSQL username
db_password = 'mypassword123'   # Replace with your PostgreSQL password
db_host = 'localhost'
db_port = '5432'
engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')


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
        def remove_accents(text):
            if not text:
                return ''
            normalized_text = unicodedata.normalize('NFD', text)
            accent_removed_text = ''.join(
                char for char in normalized_text if unicodedata.category(char) != 'Mn'
            )
            return unicodedata.normalize('NFC', accent_removed_text)

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

    def get_speech_excerpt(speech, max_length=300):
        """Create a relevant excerpt from the speech text"""
        if len(speech) <= max_length:
            return speech
        
        # Get the middle portion of the speech
        start = (len(speech) - max_length) // 2
        excerpt = speech[start:start + max_length]
        
        # Ensure we don't cut words in the middle
        first_space = excerpt.find(' ')
        last_space = excerpt.rfind(' ')
        
        return '...' + excerpt[first_space:last_space].strip() + '...'

    @app.route('/')
    def home():
        return render_template('home.html')

    @app.route('/search')
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
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
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
            
            # Fetch speeches from database using SQLAlchemy
            sql_query = text("""
                SELECT id, speech, member_name, sitting_date , political_party,parliamentary_period,parliamentary_session,parliamentary_sitting
                FROM speeches 
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
                    'speech': get_speech_excerpt(speech['speech']),
                    'member_name': speech['member_name'] or 'Unknown',
                    'political_party': speech['political_party'],
                    'score': result['similarity_score'],
                    'parliamentary_period': speech['parliamentary_period'],  # Ensure these keys are added
                    'parliamentary_session': speech['parliamentary_session'],
                    'parliamentary_sitting': speech['parliamentary_sitting'],
                    'sitting_date': speech['sitting_date'] if speech['sitting_date'] else None,
                    'political_party': speech['political_party'],
                    'score': result['similarity_score']
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
        print(f"Parameters - Period: {period}, Session: {session}, Sitting: {sitting}, Date: {date}")

        try:
            # Fetch the merged speech from the merged_speeches table
            sql_query = text("""
                SELECT parliamentary_period, parliamentary_session, parliamentary_sitting, sitting_date, merged_speech
                FROM merged_speeches
                WHERE parliamentary_period = :period
                AND parliamentary_session = :session
                AND parliamentary_sitting = :sitting
                AND sitting_date = :date
            """)
            
            with engine.connect() as connection:
                result = connection.execute(sql_query, {'period': period, 'session': session, 'sitting': sitting, 'date': date})
                merged_speech = result.fetchone()

            if merged_speech:
                return render_template('merged_speech.html', speech=merged_speech)
            else:
                return render_template('merged_speech.html', error="No merged speech found for the provided parameters.")
        except Exception as e:
            print(f"Error fetching merged speech: {e}")
            return render_template('merged_speech.html', error="An error occurred while retrieving the merged speech.")