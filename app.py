import streamlit as st
import pickle
import pandas as pd
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.markdown("""
<style>

 /* -------------------------------------- */
 /* GLOBAL BACKGROUND & TEXT COLOR FIX     */
 /* -------------------------------------- */
html, body, .stApp {
    background-color: #000 !important;
    color: #ffffff !important;
}

/* Remove all Streamlit white panels */
section.main > div, .stMarkdown, .stSelectbox, .stTextArea, .stRadio {
    background-color: transparent !important;
}

/* -------------------------------------- */
/* TITLE                                  */
/* -------------------------------------- */
.main-title {
    text-align: center;
    color: #ffffff !important;
    font-size: 52px;
    font-weight: 800;
    margin-top: -15px;
    margin-bottom: 25px;
}

/* -------------------------------------- */
/* RADIO LABELS (By Movie / By Description) */
/* -------------------------------------- */
div[data-testid="stRadio"] div[role="radiogroup"] * {
    color: #ffffff !important;   /* PURE WHITE */
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* Label above radio */
.stRadio > label {
    color: #ffffff !important;
}

/* -------------------------------------- */
/* SELECTBOX LABEL + ALL FIELD LABELS     */
/* -------------------------------------- */
div[data-testid="stSelectbox"] > label {
    color: #ffffff !important;
    font-size: 20px !important;
    font-weight: 700 !important;
}

label, .stTextInput label, .stTextArea label {
    color: #ffffff !important;
}

/* -------------------------------------- */
/* INPUT TEXTBOX                          */
/* -------------------------------------- */
textarea, input, select {
    background-color: #111 !important;
    color: #ffffff !important;
    border: 1px solid #555 !important;
}

/* -------------------------------------- */
/* BUTTON FIX (Recommend)                 */
/* -------------------------------------- */
button[kind="primary"],
.stButton>button {
    background-color: #9333ea !important;
    color: white !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease-in-out;
}

button[kind="primary"]:hover,
.stButton>button:hover {
    background-color: #b84cff !important;
    transform: scale(1.05);
}

/* -------------------------------------- */
/* MOVIE TITLES                           */
/* -------------------------------------- */
.movie-name {
    color: #ffffff !important;  /* PURE WHITE */
    font-size: 16px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 8px;
}

/* -------------------------------------- */
/* POSTER HOVER EFFECT                    */
/* -------------------------------------- */
.movie-poster img {
    transition: 0.3s ease-in-out;
    border-radius: 12px;
}

.movie-poster img:hover {
    transform: scale(1.10);
    box-shadow: 0px 6px 30px rgba(147, 51, 234, 0.9);
    border: 2px solid #9333ea;
    cursor: pointer;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)

# TAG TOKENIZATION

def _to_tokens(s):
    if pd.isna(s):
        return set()
    s = str(s).lower()
    for sep in ['|', ',', ';']:
        s = s.replace(sep, ' ')
    return set(t for t in s.split() if t)


def compute_tag_relevance(movies_df, base_idx, candidate_indices):
    base_tokens = _to_tokens(movies_df.iloc[base_idx]['tags'])
    scores = []
    for idx in candidate_indices:
        other_tokens = _to_tokens(movies_df.iloc[idx]['tags'])
        inter = len(base_tokens & other_tokens)
        union = len(base_tokens | other_tokens) or 1
        scores.append(inter / union)
    return scores

# NDCG
def ndcg_at_k(relevance_scores, k):
    rel = relevance_scores[:k]
    ideal = sorted(relevance_scores, reverse=True)
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return (dcg / idcg) if idcg != 0 else 0.0

# LOAD DATA
movies = pickle.load(open('artificats/movie_list.pkl', 'rb'))
similarity = pickle.load(open('artificats/similarity.pkl', 'rb'))

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['tags'])

TMDB_API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# FETCH POSTER
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

# MOVIE RECOMMENDER
def recommend_movie(movie_title):
    movie_index = movies[movies['title'].str.lower() == movie_title.lower()].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_posters = []

    for idx, _ in movie_list:
        movie_id = movies.iloc[idx].movie_id
        recommended_movies.append(movies.iloc[idx].title)
        recommended_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_posters, movie_list


option = st.radio("Choose recommendation type:", ('By Movie', 'By Description'))



# MOVIE-BASED SEARCH
if option == 'By Movie':
    selected_movie = st.selectbox("Select a movie you like:", movies['title'].values)

    if st.button("Recommend"):
        recommended_movies, recommended_posters, movie_list = recommend_movie(selected_movie)

        movie_index = movies[movies['title'].str.lower() == selected_movie.lower()].index[0]
        top_indices = [i[0] for i in movie_list]
        rel_scores = compute_tag_relevance(movies, movie_index, top_indices)
        ndcg_score = ndcg_at_k(rel_scores, k=5)

        # SHOW NDCG on UI
        st.write(f"📊 **NDCG@5 (tag-based evaluation): {ndcg_score:.4f}**")

        cols = st.columns(len(recommended_movies))
        for i in range(len(recommended_movies)):
            with cols[i]:
                st.markdown(f"<div class='movie-name'>{recommended_movies[i]}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div class="movie-poster">
                        <img src="{recommended_posters[i]}" style="width:100%;">
                    </div>
                    """,
                    unsafe_allow_html=True
                )
# DESCRIPTION-BASED SEARCH
elif option == 'By Description':
    description = st.text_area("Describe the kind of movie you want to watch:")

    if st.button("Recommend"):
        # TF-IDF similarity
        user_tfidf = tfidf.transform([description])
        cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)

        # Get TOP-5 only
        similar_indices = cosine_sim[0].argsort()[-6:][::-1]  # 1 base + 5 recommendations

        # Extract movies/posters
        recommended_movies = movies.iloc[similar_indices]['title'].values
        recommended_posters = [fetch_poster(movies.iloc[i].movie_id) for i in similar_indices]

        # NDCG calculation
        base_idx = similar_indices[0]
        candidates = [i for i in similar_indices if i != base_idx]

        rel_scores = compute_tag_relevance(movies, base_idx, candidates)
        ndcg_score = ndcg_at_k(rel_scores, k=min(5, len(candidates)))

        # SHOW NDCG on UI
        st.write(f"📊 **NDCG@5 (tag-based evaluation): {ndcg_score:.4f}**")

        # Display only TOP-5 results
        num_results = min(len(recommended_movies), 5)

        cols = st.columns(5)
        for idx in range(num_results):
            with cols[idx]:
                st.markdown(
                    f"<p class='movie-name'>{recommended_movies[idx]}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<div class='movie-poster'><img src='{recommended_posters[idx]}' width='250'></div>",
                    unsafe_allow_html=True
                )

