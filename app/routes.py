from flask import request, render_template
from app import db
from app.models import Speech
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load vectorizerr and TF-IDF matrix at the module level
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

with open('speech_ids.pkl', 'rb') as f:
    speech_ids = pickle.load(f)


def create_routes(app):
    @app.route('/', methods=['GET', 'POST'])
    def search():
        query = request.args.get('q', '')

        results = []
        if query:
           
        
            # Transform the query using the same vectorizer
            query_tfidf = vectorizer.transform([query])
            
            # Compute cosine similarity between the query and all speeches
            similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
            
            # Get top k results (e.g., top 10)
            top_k = 10
            top_indices = similarity_scores.argsort()[-top_k:][::-1]
            top_scores = similarity_scores[top_indices]
            top_speech_ids = [speech_ids[i] for i in top_indices]
            
            # Fetch the corresponding speeches from the database
            top_speeches = Speech.query.filter(Speech.id.in_(top_speech_ids)).all()
            
            # Create a dictionary to map speech ID to similarity score
            score_dict = {id_: score for id_, score in zip(top_speech_ids, top_scores)}
            
            # Prepare results
            for speech in top_speeches:
                results.append({
                    'member_name': speech.member_name,
                    'sitting_date': speech.sitting_date,
                    'speech': (speech.speech[:200] if speech.speech else ''),  # Display first 200 characters if not None
                    'score': score_dict[speech.id]
                })

            # Sort results by score in descending order
            results = sorted(results, key=lambda x: x['score'], reverse=True)

        return render_template('search.html', query=query, results=results)
