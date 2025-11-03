# app.py
import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ----------------------------
# Ensure artificats folder exists
# ----------------------------
os.makedirs("artificats", exist_ok=True)

# ----------------------------
# Download large files if missing
# ----------------------------
def download_file(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

# Replace these URLs with your own uploaded files (GitHub, Google Drive, or S3)
MOVIE_LIST_URL = "https://raw.githubusercontent.com/Akanksha25300/movie-recommender-system/main/artificats/movie_list.pkl"
SIMILARITY_URL = "https://raw.githubusercontent.com/Akanksha25300/movie-recommender-system/main/artificats/similarity.pkl"

download_file(MOVIE_LIST_URL, "artificats/movie_list.pkl")
download_file(SIMILARITY_URL, "artificats/similarity.pkl")

# ----------------------------
# Load pickled data
# ----------------------------
movies = pickle.load(open('artificats/movie_list.pkl', 'rb'))
similarity = pickle.load(open('artificats/similarity.pkl', 'rb'))

# Pre-calc TF-IDF for description based search
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

# TMDB API Key
TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# ----------------------------
# Fetch Poster Utility
# ----------------------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        pass
    return "https://placehold.co/500x750?text=No+Image"

# ----------------------------
# Movie Based Recommendation
# ----------------------------
def recommend_movie(movie_title):
    movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]  # ✅ Only 5 movies

    recommended_movies = []
    recommended_posters = []

    for i in movie_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")

option = st.radio("Choose recommendation type:", ('By Movie', 'By Description'))

# ✅ Movie-Based Search → 5 movies only
if option == 'By Movie':
    selected_movie = st.selectbox("Select a movie you like:", movies['title'].values)

    if st.button("Recommend"):
        recommended_movies, recommended_posters = recommend_movie(selected_movie)
        num_results = len(recommended_movies)

        cols = st.columns(num_results)
        for i in range(num_results):
            with cols[i]:
                st.text(recommended_movies[i])
                st.image(recommended_posters[i])

# ✅ Description-Based Search → 10 movies (5 + 5)
elif option == 'By Description':
    description = st.text_area("Describe the kind of movie you want to watch:")

    if st.button("Recommend"):
        user_tfidf = tfidf.transform([description])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
        similar_indices = cosine_sim[0].argsort()[-10:][::-1]

        recommended_movies = movies.iloc[similar_indices]['title'].values
        recommended_posters = [fetch_poster(movies.iloc[i].movie_id) for i in similar_indices]

        num_results = min(len(recommended_movies), 10)

        for start in range(0, num_results, 5):
            cols = st.columns(5)
            for idx in range(start, min(start + 5, num_results)):
                with cols[idx - start]:
                    st.text(recommended_movies[idx])
                    st.image(recommended_posters[idx])
