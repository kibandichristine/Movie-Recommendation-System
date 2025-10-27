import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import joblib

# Load saved artifacts
# Note: For large matrices, loading directly into Streamlit's cache is ideal.
@st.cache_data
def load_assets():
    movie_matrix = pd.read_pickle('movie_matrix.pkl')
    user_similarity = np.load('user_similarity.npy')
    item_similarity = np.load('item_similarity.npy')
    svd_model = joblib.load('svd_model.joblib')
    return movie_matrix, user_similarity, item_similarity, svd_model

# ... then call the function in your Streamlit app:
# movie_matrix, user_similarity, item_similarity, svd_model = load_assets()
# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="MovieLens Puppet Theater Recommender",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling Function ---
def set_background_and_style():
    """Injects custom CSS for background gradient, puppet theater theme, and content styling."""
    
    # Dark Mode (Black) with Royal Blue and Gold Themed Styling
    custom_css = """
    <style>
    /* Pure Black Background for the main page */
    .stApp {
        background-color: #000000; 
        /* Removed gradient to ensure pure black */
        background-image: none;
        color: #000000; /* Light text for general visibility */
    }
    
    /* Style for the main content container (Simulating a stage/card effect) */
    .stContainer {
        padding: 30px 40px; /* Increased padding */
        /* Very dark, semi-transparent background to fit the dark theme */
        background-color: rgba(0, 0, 0, 0.85); 
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(255, 255, 255, 0.1); /* Lighter shadow for dark background */
        margin-top: 20px;
        
        /* Puppet Show/Theater Border Effect - Using Gold Accent */
        border: 4px dashed #FFC300; 
        color: #000000; /* Default text color is now light gray */
    }

    /* Make ALL text inside the container light for visibility against dark container background */
    .stContainer * {
        color: #F0F0F0 !important;
    }
    
    /* Overriding specific Streamlit text elements that might remain dark */
    /* Selectbox/slider labels are now explicitly set to light gray */
    .stSelectbox label, .stSlider label {
        color: #F0F0F0 !important;
    }

    /* Style for the Streamlit info/success boxes */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        color: #333333 !important; /* Ensure internal text is dark for contrast */
        background-color: #F0F0F0 !important;
    }
    .st-emotion-cache-1c9vcm4{
        color: #333333 !important;
    }
    

    /* Adjust Streamlit primary-colored buttons for visual harmony and size */
    .stButton>button {
        border-radius: 12px;
        border: 2px solid #FFC300; /* Gold/Yellow border */
        background-color: #6A1B9A; /* Deep Purple accent */
        color: #FFFFFF; /* Keep button text white for contrast on purple */
        font-size: 18px; /* Slightly larger text */
        padding: 10px 20px; /* Bigger button size */
        transition: transform 0.1s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px); /* Subtle lift effect on hover */
        box-shadow: 0 2px 5px rgba(255, 255, 255, 0.4);
    }
    
    /* Styling for the main titles */
    h1 {
        color: #FFC300 !important; /* Gold text for prominence */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.8);
    }
    h3, h2, .subheader {
        color: #4169E1 !important; /* Royal Blue for secondary headers */
    }

    /* Fix sidebar elements for readability against the dark background */
    .css-1e6z71b { /* Targeting radio button group container */
        background-color: #1A1A1A;
        border-radius: 10px;
        padding: 10px;
    }
    .css-1e6z71b * {
        color: #F0F0F0 !important; /* Light text in sidebar */
    }
    
    /* Ensure markdown text in the sidebar is also light */
    .stSidebar .stMarkdown * {
        color: #F0F0F0 !important;
    }
    

    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

# --- Caching Data Loading and Model Preprocessing ---
@st.cache_data
def load_and_process_data():
    """Loads data, creates the user-item matrix, and calculates similarity matrices."""
    try:
        # NOTE: Assumes 'movies.csv' and 'ratings.csv' are in the same directory
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        st.error("Error: CSV files ('movies.csv', 'ratings.csv') not found. Please ensure they are uploaded or available.")
        return None, None, None, None, None, None

    # Merge dataframes (as done in the notebook)
    df = pd.merge(movies, ratings, on='movieId')

    # Data Cleaning and Preparation (from notebook)
    df['title_raw'] = df['title']
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
    df.dropna(subset=['rating'], inplace=True)
    
    # Pre-merge movie genres back for UI context later
    movie_genres = movies[['title', 'genres']].copy()
    movie_genres['title'] = movie_genres['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
    movie_genres = movie_genres.drop_duplicates(subset=['title'])
    
    # Create the User-Item Matrix (Pivot Table)
    movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    movie_matrix_sparse = csr_matrix(movie_matrix.values)
    
    # Calculate Similarity Matrices (core models)
    user_similarity = cosine_similarity(movie_matrix_sparse)
    item_similarity = cosine_similarity(movie_matrix_sparse.T)

    return df, movie_matrix, movie_matrix_sparse, user_similarity, item_similarity, movie_genres

# --- Recommendation Functions ---

def recommend_movies_usercf(user_id, movie_matrix, user_similarity, n=10):
    """Recommend movies using User-Based Collaborative Filtering."""
    user_index = movie_matrix.index.get_loc(user_id)
    
    # Calculate predicted ratings for all movies for this user
    # Scores = user_similarity @ movie_matrix
    scores = user_similarity[user_index] @ movie_matrix.values
    
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

# 0. Apply Custom Styling
set_background_and_style()


# 1. Load Data
load_result = load_and_process_data()
if load_result is None:
    st.stop()
    
df_raw, movie_matrix, movie_matrix_sparse, user_similarity, item_similarity, movie_genres = load_result

# Place main content inside a themed container
with st.container():
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Deployment of Collaborative Filtering Model")
    st.info("Welcome to the puppet show! Select a model in the sidebar to begin. The similarity calculation is cached for fast performance.")

    # Sidebar for Configuration
    st.sidebar.header("⚙️ Configuration")
    model_type = st.sidebar.radio("Select Recommendation Model", 
                                  ["User-Based CF", "Item-Based CF", "Popularity"])
    N_RECOMMENDATIONS = st.sidebar.slider("Number of Recommendations", 5, 20, 10)


    # --- Main Content Area Logic ---

    if model_type == "User-Based CF":
        st.header("👤 1. User-Based Collaborative Filtering")
        st.markdown("Recommends movies based on the tastes of similar users.")
        
        user_ids = movie_matrix.index.tolist()
        rating_counts_per_user = df_raw.groupby('userId')['rating'].count()
        default_user = rating_counts_per_user.sort_values(ascending=False).index[0]
        
        selected_user = st.selectbox("Select a User ID", user_ids, index=user_ids.index(default_user))
        
        if st.button(f"✨ Pull the Strings! Get Recommendations for User {selected_user}", key="user_cf_button"):
            with st.spinner(f"Finding top {N_RECOMMENDATIONS} movies for User {selected_user}..."):
                try:
                    recommendations = recommend_movies_usercf(
                        selected_user, movie_matrix, user_similarity, N_RECOMMENDATIONS
                    )
                    
                    st.subheader("💡 Top Recommended Movies")
                    
                    if not recommendations.empty:
                        rec_df = recommendations.reset_index()
                        rec_df.columns = ["Movie Title", "Predicted Score"]
                        rec_df['Predicted Score'] = rec_df['Predicted Score'].round(2)
                        
                        st.dataframe(rec_df, use_container_width=True)
                        st.success("The show is a hit! Recommendations generated successfully!")
                    else:
                        st.warning("The user has already seen everything! No new recommendations found.")

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    
    elif model_type == "Item-Based CF":
        st.header("🎥 2. Item-Based Collaborative Filtering")
        st.markdown("Recommends movies similar to one you already like.")
        
        movie_titles = movie_matrix.columns.tolist()
        
        selected_movie = st.selectbox("Select a Movie to get similar recommendations", movie_titles)

        # Display genre information for context in the sidebar
        selected_genre = movie_genres[movie_genres['title'] == selected_movie]['genres'].iloc[0]
        st.sidebar.markdown(f"**Selected Movie Genre:**")
        st.sidebar.markdown(f"*{selected_genre}*")
        
        if st.button(f"🔍 Find Similar Movies for '{selected_movie}'", key="item_cf_button"):
            with st.spinner(f"Finding top {N_RECOMMENDATIONS} similar movies..."):
                try:
                    recommendations = recommend_movies_itemcf(
                        selected_movie, movie_matrix, item_similarity, N_RECOMMENDATIONS
                    )
                    
                    st.subheader(f"🤝 Top {N_RECOMMENDATIONS} Similar Movies")
                    
                    if not recommendations.empty:
                        rec_df = recommendations.reset_index()
                        rec_df.columns = ["Movie Title", "Similarity Score"]
                        rec_df['Similarity Score'] = rec_df['Similarity Score'].round(4)
                        
                        st.dataframe(rec_df, use_container_width=True)
                        st.success("The curtain rises! Similar movies found.")
                    else:
                        st.warning("No similar movies found in the repertoire.")
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    elif model_type == "Popularity":
        st.header("⭐ 3. Popularity-Based Recommendations")
        st.markdown("Recommendations for new users based on overall popularity and average rating (requires 50+ ratings).")

        if st.button(f"🎭 Show Top Popular Picks", key="pop_cf_button"):
            with st.spinner(f"Calculating top {N_RECOMMENDATIONS} popular movies..."):
                try:
                    popular_df = get_most_popular_movies(df_raw, N_RECOMMENDATIONS)
                    
                    st.subheader("📈 The Marquee Hits (Most Popular)")
                    
                    if not popular_df.empty:
                        # Display metrics for the absolute top movie
                        top_movie = popular_df.iloc[0]
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Absolute Top Movie", top_movie.name)
                        col2.metric("Average Rating", f"{top_movie['Average Rating']:.2f} / 5.0")
                        col3.metric("Total Ratings", int(top_movie['Rating Count']))
                        
                        # Display the rest of the list
                        st.dataframe(popular_df, use_container_width=True)
                        st.success("The crowd has spoken! Popularity recommendations ready.")
                    else:
                        st.warning("Could not calculate popular movies. Ensure sufficient ratings data.")
                        
                except Exception as e:
                    st.error(f"An error occurred while fetching popular movies: {e}")

    st.markdown("---")
    st.caption("A cinematic experience brought to you by data. ")
