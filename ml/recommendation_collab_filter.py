import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict


# --- User-Based Collaborative Filtering ---
def get_user_recommendations(
    user_id: str,
    user_item_matrix: pd.DataFrame,  # The original matrix with NaNs
    n: int = 3,
) -> list[tuple[str, float]]:
    """
    Generates top-N recommendations for a user based on user-user similarity.

    Args:
        user_id: The ID of the user to generate recommendations for.
        user_item_matrix: DataFrame with users as rows, items as columns, and ratings as values (can contain NaNs).
        n: The number of recommendations to return.

    Returns:
        A list of top-N recommended items and their predicted scores.
    """
    # --- Calculate User-User Similarity ---
    # This logic is moved here from the main function for better modularity.
    # Fill NaNs with 0 for similarity calculation.
    user_item_matrix_filled = user_item_matrix.fillna(0)
    # Calculate cosine similarity between all pairs of users.
    user_similarity = cosine_similarity(user_item_matrix_filled)
    user_similarity_df = pd.DataFrame(
        user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    # 1. Get similarity scores for the target user and sort them
    # This series will have other users as the index and their similarity to `user_id` as values.
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # 2. Get items the target user has already rated
    # We use .dropna() to filter out items with no rating (NaN) and get their index (item names).
    items_rated_by_user = user_item_matrix.loc[user_id].dropna().index

    # 3. Calculate weighted scores for unrated items
    item_scores = {}
    # Iterate through all possible items in the dataset.
    for item in user_item_matrix.columns:
        # We only want to predict scores for items the user has NOT rated.
        if item not in items_rated_by_user:
            weighted_sum = 0
            similarity_sum = 0
            # Iterate through all other users and their similarity to the target user.
            # `similar_users` is already sorted, so we are implicitly considering more similar users first.
            for other_user, similarity in similar_users.items():
                # We don't want to include the target user's own (non-existent) ratings.
                # We only consider other users who have actually rated the item in question.
                if other_user != user_id and not pd.isna(
                    user_item_matrix.loc[other_user, item]
                ):
                    # The score is a weighted sum of ratings from other users.
                    # The weight is the similarity between the target user and the other user.
                    weighted_sum += similarity * user_item_matrix.loc[other_user, item]
                    similarity_sum += similarity
            # To get the final predicted score, we normalize the weighted sum.
            # This avoids a bias towards items rated by many users, regardless of similarity.
            if similarity_sum > 0:
                item_scores[item] = weighted_sum / similarity_sum

    # 4. Sort items by score and return top N
    # We sort the dictionary of predicted scores in descending order.
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    # Return the top `n` items and their scores.
    return sorted_items[:n]


# --- Item-Based Collaborative Filtering ---
def get_item_recommendations(
    user_id: str,
    user_item_matrix: pd.DataFrame,  # The original matrix with NaNs
    n: int = 3,
) -> list[tuple[str, float]]:
    """
    Generates top-N recommendations for a user based on item-item similarity.

    Args:
        user_id: The ID of the user to generate recommendations for.
        user_item_matrix: DataFrame with users as rows, items as columns, and ratings as values (can contain NaNs).
        n: The number of recommendations to return.

    Returns:
        A list of top-N recommended items and their predicted scores.
    """
    # --- Calculate Item-Item Similarity ---
    # This logic is moved here from the main function for better modularity.
    # Fill NaNs with 0 and transpose to calculate similarity between items.
    user_item_matrix_filled = user_item_matrix.fillna(0)
    item_similarity = cosine_similarity(user_item_matrix_filled.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns,
    )

    # 1. Get items the user has rated and their ratings
    # This gives us a series where the index is the item name and the value is the rating.
    user_ratings = user_item_matrix.loc[user_id].dropna()

    # 2. Calculate weighted scores for all items
    # Using defaultdict(float) simplifies the accumulation of scores.
    item_scores = defaultdict(float)
    # Iterate through each item the user has already rated.
    for rated_item, rating in user_ratings.items():
        # For each rated item, get a series of other items and their similarity scores.
        similar_items = item_similarity_df[rated_item].sort_values(ascending=False)
        # Iterate through the items similar to the one the user has rated.
        for similar_item, similarity in similar_items.items():
            # We only want to recommend items the user has NOT already rated.
            if similar_item not in user_ratings.index:
                # The score for a potential recommendation is based on how similar it is
                # to items the user has already liked.
                # The score is weighted by the user's rating for the original item.
                # A high rating for `rated_item` gives more weight to items similar to it.
                item_scores[similar_item] += rating * similarity

    # 3. Sort items by score and return top N
    # Sort the calculated scores in descending order.
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    # Return the names of the top `n` recommended items and their scores.
    return sorted_items[:n]


def main():
    """
    Main function to demonstrate user-based and item-based collaborative filtering.
    """
    # --- 1. Sample Data ---
    # A simple user-item matrix where rows are users, columns are movies,
    # and values are ratings (1-5). NaN means the user hasn't rated the movie.
    # This is a sparse matrix, which is more typical of real-world datasets.
    data = {
        "User A": {  # Target user for recommendations.
            "Movie 1": 5,
            "Movie 2": None,
            "Movie 3": 4,
            "Movie 4": 4,
            "Movie 5": None,
        },
        "User B": {
            "Movie 1": 3,
            "Movie 2": None,
            "Movie 3": None,
            "Movie 4": None,
            "Movie 5": 3,
        },
        "User C": {
            "Movie 1": 4,
            "Movie 2": 3,
            "Movie 3": 4,
            "Movie 4": None,
            "Movie 5": 5,
        },
        "User D": {
            "Movie 1": None,
            "Movie 2": None,
            "Movie 3": None,
            "Movie 4": 5,
            "Movie 5": 4,
        },
        "User E": {
            "Movie 1": None,
            "Movie 2": 5,
            "Movie 3": 5,
            "Movie 4": None,
            "Movie 5": None,
        },
    }
    # Create a pandas DataFrame from the dictionary.
    # The initial DataFrame has users as columns and movies as the index.
    # .T transposes the DataFrame, making users the rows and movies the columns,
    # which is the standard "user-item matrix" format.
    user_item_matrix = pd.DataFrame(data).T
    print("--- User-Item Ratings Matrix ---")
    print(user_item_matrix)

    # --- 2. User-Based Collaborative Filtering ---
    print("\n--- User-Based Recommendations ---")

    # Get recommendations for User A
    # The similarity calculation is now encapsulated within the function.
    user_recs = get_user_recommendations("User A", user_item_matrix)
    print(f"\nUser-Based Recommendations for User A:")
    if user_recs:
        for item, score in user_recs:
            print(f"  - {item}: Predicted Score = {score:.4f}")
    else:
        print("  - No recommendations found.")

    # --- 3. Item-Based Collaborative Filtering ---
    print("\n--- Item-Based Recommendations ---")

    # Get recommendations for User A
    # The similarity calculation is now encapsulated within the function.
    item_recs = get_item_recommendations("User A", user_item_matrix)
    print(f"\nItem-Based Recommendations for User A:")
    if item_recs:
        for item, score in item_recs:
            print(f"  - {item}: Predicted Score = {score:.4f}")
    else:
        print("  - No recommendations found.")


# This is a standard Python construct. The code inside this block will only run
# when the script is executed directly (not when it's imported as a module).
if __name__ == "__main__":
    main()
