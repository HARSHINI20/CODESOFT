import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load datasets
# Assuming 'ratings.csv' has columns: 'userId', 'movieId', 'rating'
ratings_data = pd.read_csv("C:\\Users\\harshini\\Desktop\\recomandation system\\ratings.csv")

# Assuming 'movies.csv' has columns: 'movieId', 'title'
movies_data = pd.read_csv("C:\\Users\\harshini\\Desktop\\recomandation system\\movies.csv")
# Step 2: Merge datasets based on 'movieId'
merged_data = pd.merge(ratings_data, movies_data, on='movieId')

# Step 3: Data Preprocessing
user_movie_matrix = merged_data.pivot_table(index='userId', columns='movieId', values='rating')

# Step 4: Similarity Calculation
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))


# Step 5: Recommendation Generation
def get_recommendations(userId, user_movie_matrix, user_similarity):
    # Find similar users for the given userId
    similar_users = user_similarity[userId - 1]  # Assuming user indexing starts from 1

    # Get movies rated by the given user
    user_rated_movies = user_movie_matrix.loc[userId].dropna().index

    recommended_movies = []
    for similar_user_id, similarity_score in enumerate(similar_users):
        if similarity_score > 0 and similar_user_id + 1 != userId:  # Exclude the given user
            similar_user_rated_movies = user_movie_matrix.loc[similar_user_id + 1].dropna().index
            # Recommend movies that similar users rated highly and the given user hasn't seen
            recommendations = set(similar_user_rated_movies) - set(user_rated_movies)
            recommended_movies.extend(recommendations)

    # Get movie names corresponding to movieIds
    movie_names = movies_data[movies_data['movieId'].isin(recommended_movies)]['title'].unique()

    return list(movie_names)[:10]  # Return top 10 recommendations with movie names


# Get recommendations for user 5
user_5_recommendations = get_recommendations(5, user_movie_matrix, user_similarity)
print("Recommended movies for User 5:", user_5_recommendations)