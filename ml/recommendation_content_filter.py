import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


def generate_recommendations_from_embeddings(
    movies_df: pd.DataFrame, user_ratings: dict
) -> pd.Series:
    """
    Generates movie recommendations for a user based on content embeddings.

    Args:
        movies_df: DataFrame containing movie titles and genres.
        user_ratings: A dictionary of the user's past ratings.

    Returns:
        A pandas Series of recommended movies and their scores.
    """
    # --- 1. Simulate Pre-trained Word Vectors & Define Helper ---
    vector_dim = 5
    word_vectors = {
        # Sci-Fi / Action
        "matrix": np.array([0.9, 0.2, -0.1, 0.1, 0.3]),
        "inception": np.array([0.8, 0.3, -0.2, 0.2, 0.2]),
        "blade": np.array([0.7, 0.1, 0.1, 0.1, 0.4]),
        "runner": np.array([0.6, 0.2, 0.2, 0.1, 0.3]),
        "alien": np.array([0.4, -0.2, -0.8, 0.9, 0.1]),
        "action": np.array([0.5, -0.5, 0.8, 0.3, 0.5]),
        "sci": np.array([0.9, 0.5, 0.1, 0.1, 0.2]),
        "fi": np.array([0.9, 0.5, 0.1, 0.1, 0.2]),  # Treat "sci-fi" as two words
        "thriller": np.array([0.2, -0.6, 0.7, 0.8, 0.6]),
        "horror": np.array([0.1, -0.8, -0.6, 0.9, 0.2]),
        # Crime / Drama
        "godfather": np.array([-0.8, 0.9, 0.3, -0.5, -0.7]),
        "pulp": np.array([-0.5, 0.6, 0.5, -0.4, -0.5]),
        "fiction": np.array([-0.4, 0.5, 0.6, -0.3, -0.6]),
        "crime": np.array([-0.9, 0.8, 0.4, -0.6, -0.8]),
        "drama": np.array([-0.2, 0.7, -0.5, -0.8, -0.4]),
        # Romance
        "forrest": np.array([-0.1, 0.8, -0.9, -0.9, -0.1]),
        "gump": np.array([-0.2, 0.7, -0.8, -0.8, -0.2]),
        "romance": np.array([0.1, 0.9, -0.7, -0.9, 0.1]),
    }

    # --- 4. Create Item Embeddings ---
    # Combine title and genres into a single content string for each movie.
    movies_df["Content"] = movies_df["Title"] + " " + movies_df["Genres"]

    # Generate a single vector (embedding) for each movie.
    item_embeddings = np.array(
        [
            get_item_vector(content, word_vectors, vector_dim)
            for content in movies_df["Content"]
        ]
    )

    # --- 5. Build User Profile ---
    # This logic is identical to the TF-IDF version, but uses the new embeddings.
    rated_indices = movies_df[movies_df["Title"].isin(user_ratings.keys())].index
    rated_vectors = item_embeddings[rated_indices]
    user_profile_ratings = np.array(
        [user_ratings[title] for title in movies_df.loc[rated_indices, "Title"]]
    )

    user_profile = np.dot(user_profile_ratings, rated_vectors)
    user_profile = user_profile.reshape(1, -1)

    # --- 6. Generate Recommendations ---
    recommendation_scores = cosine_similarity(user_profile, item_embeddings)
    scores_series = pd.Series(recommendation_scores.flatten(), index=movies_df["Title"])

    seen_movies = user_ratings.keys()
    recommendations = scores_series.drop(seen_movies)
    top_recommendations = recommendations.sort_values(ascending=False)

    print("\n--- Recommendations for the User (based on embeddings) ---")
    print(top_recommendations)


if __name__ == "__main__":
    main()
