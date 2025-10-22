import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="MovieLens Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Data Loading and Model Preprocessing ---
# This function performs the heavy lifting and is cached,
# so it only runs once when the app is launched.
@st.cache_data
def load_and_process_data():
    """Loads data, creates the user-item matrix, and calculates similarity matrices."""
    try:
        # NOTE: Assumes 'movies.csv' and 'ratings.csv' are in the same directory
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        st.error("Error: CSV files ('movies.csv', 'ratings.csv') not found. Please ensure they are uploaded or available.")
        return None, None, None, None, None

    # Merge dataframes (as done in the notebook)
    df = pd.merge(movies, ratings, on='movieId')

    # Data Cleaning and Preparation (from notebook)
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
    df.dropna(subset=['rating'], inplace=True)

    # Create the User-Item Matrix (Pivot Table)
    movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)
    
    # Calculate Similarity Matrices (core models)
    user_similarity = cosine_similarity(movie_matrix_sparse)
    item_similarity = cosine_similarity(movie_matrix_sparse.T)

    return df, movie_matrix, movie_matrix_sparse, user_similarity, item_similarity

# --- Recommendation Functions ---

def recommend_movies_usercf(user_id, movie_matrix, user_similarity, n=10):
    """Recommend movies using User-Based Collaborative Filtering."""
    user_index = movie_matrix.index.get_loc(user_id)
    
    # Calculate predicted ratings for all movies for this user
    # Scores = user_similarity @ movie_matrix
    scores = user_similarity[user_index] @ movie_matrix.values
    
    # Create a Series for easier handling
    scores_series = pd.Series(scores, index=movie_matrix.columns)
    
    # Get movies the user has already rated (exclude them)
    rated_movies = movie_matrix.iloc[user_index][movie_matrix.iloc[user_index] > 0].index
    scores_series = scores_series.drop(rated_movies, errors='ignore')
    
    # Return top N recommendations
    recommendations = scores_series.sort_values(ascending=False).head(n)
    return recommendations

def recommend_movies_itemcf(movie_title, movie_matrix, item_similarity, n=10):
    """Recommend similar movies using Item-Based Collaborative Filtering."""
    try:
        movie_idx = movie_matrix.columns.get_loc(movie_title)
        
        # Get similarity scores for the selected movie
        similar_movies = item_similarity[movie_idx]
        
        # Create a Series and exclude the movie itself
        similar_movies_series = pd.Series(similar_movies, index=movie_matrix.columns)
        similar_movies_series = similar_movies_series.drop(movie_title, errors='ignore')
        
        # Return top N recommendations
        recommendations = similar_movies_series.sort_values(ascending=False).head(n)
        return recommendations
    except KeyError:
        st.error(f"Movie '{movie_title}' not found in the dataset.")
        return pd.Series()

def get_most_popular_movies(df, n=10):
    """Returns top N movies based on average rating (filtered by count)."""
    # Filter for movies with at least 50 ratings to ensure popularity is meaningful
    movie_counts = df.groupby('title')['rating'].count()
    popular_movies = movie_counts[movie_counts >= 50].index
    
    popular_df = df[df['title'].isin(popular_movies)]
    
    # Group by title and sort by mean rating
    avg_ratings = popular_df.groupby('title')['rating'].mean()
    rating_counts = popular_df.groupby('title')['rating'].count()
    
    final_df = pd.DataFrame({
        'Average Rating': avg_ratings,
        'Rating Count': rating_counts
    }).sort_values(by=['Average Rating', 'Rating Count'], ascending=False)
    
    return final_df.head(n)


# --- Streamlit UI Components ---

# 1. Load Data
df_raw, movie_matrix, movie_matrix_sparse, user_similarity, item_similarity = load_and_process_data()

if df_raw is None:
    st.stop()


st.title("🎬 Movie Recommendation System")
st.markdown("### Deployment of Collaborative Filtering Model")
st.info("The application automatically loads your `ratings.csv` and `movies.csv` and calculates the similarity matrices. This heavy computation is cached for fast performance!")

# Sidebar for Configuration
st.sidebar.header("Configuration")
model_type = st.sidebar.radio("Select Recommendation Model", 
                              ["User-Based CF", "Item-Based CF", "Popularity"])
N_RECOMMENDATIONS = st.sidebar.slider("Number of Recommendations", 5, 20, 10)


# --- Main Content Area ---

if model_type == "User-Based CF":
    st.header("1. User-Based Collaborative Filtering")
    st.markdown("Recommends movies based on the tastes of similar users.")
    
    user_ids = movie_matrix.index.tolist()
    # Find a user with a good number of ratings for demonstration
    rating_counts_per_user = df_raw.groupby('userId')['rating'].count()
    
    # Default to a user with high activity
    default_user = rating_counts_per_user.sort_values(ascending=False).index[0]
    
    selected_user = st.selectbox("Select a User ID", user_ids, index=user_ids.index(default_user))
    
    if st.button(f"Get Recommendations for User {selected_user}", key="user_cf_button"):
        with st.spinner(f"Finding top {N_RECOMMENDATIONS} movies for User {selected_user}..."):
            try:
                # Use the pre-calculated data
                recommendations = recommend_movies_usercf(
                    selected_user, movie_matrix, user_similarity, N_RECOMMENDATIONS
                )
                
                st.subheader("Top Recommended Movies")
                
                if not recommendations.empty:
                    rec_df = recommendations.reset_index()
                    rec_df.columns = ["Movie Title", "Predicted Score"]
                    # Predicted score here is the similarity-weighted average, let's normalize it for display
                    rec_df['Predicted Score'] = rec_df['Predicted Score'].round(2)
                    
                    st.dataframe(rec_df, use_container_width=True)
                    st.success("Recommendations generated successfully!")
                else:
                    st.warning("Could not generate recommendations. The user may have rated almost all movies.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                
elif model_type == "Item-Based CF":
    st.header("2. Item-Based Collaborative Filtering")
    st.markdown("Recommends movies similar to one you already like.")
    
    movie_titles = movie_matrix.columns.tolist()
    
    selected_movie = st.selectbox("Select a Movie to get similar recommendations", movie_titles)
    
    if st.button(f"Find Similar Movies for '{selected_movie}'", key="item_cf_button"):
        with st.spinner(f"Finding top {N_RECOMMENDATIONS} similar movies..."):
            try:
                # Use the pre-calculated data
                recommendations = recommend_movies_itemcf(
                    selected_movie, movie_matrix, item_similarity, N_RECOMMENDATIONS
                )
                
                st.subheader(f"Top {N_RECOMMENDATIONS} Similar Movies")
                
                if not recommendations.empty:
                    rec_df = recommendations.reset_index()
                    rec_df.columns = ["Movie Title", "Similarity Score"]
                    rec_df['Similarity Score'] = rec_df['Similarity Score'].round(4)
                    
                    st.dataframe(rec_df, use_container_width=True)
                    st.success("Recommendations generated successfully!")
                else:
                    st.warning("Could not find similar movies.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif model_type == "Popularity":
    st.header("3. Popularity-Based Recommendations")
    st.markdown("Recommendations for new users based on overall popularity and average rating.")

    with st.spinner(f"Calculating top {N_RECOMMENDATIONS} popular movies..."):
        try:
            popular_df = get_most_popular_movies(df_raw, N_RECOMMENDATIONS)
            st.subheader("Top Popular Movies (Rated by 50+ users)")
            st.dataframe(popular_df, use_container_width=True)
            st.success("Popularity recommendations ready!")
        except Exception as e:
            st.error(f"An error occurred while fetching popular movies: {e}")

st.markdown("---")
st.caption("Developed using the MovieLens dataset and collaborative filtering techniques.")
