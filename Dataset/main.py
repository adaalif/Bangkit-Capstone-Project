from flask import Flask, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
from model import CustomAutoencoder  # Import your custom model class
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model with custom layers
try:
    # Load the pre-trained model only once when the app starts
    loaded_model = tf.keras.models.load_model('custom_autoencoder_model.keras', custom_objects={'CustomAutoencoder': CustomAutoencoder})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Placeholder for latent embeddings (this should ideally come from your dataset or model)
latent_embeddings = np.load("latent_embeddings.npy")  # Replace with actual latent embeddings file

# Load the dataset (replace with your actual dataset)
recipes_df = pd.read_csv("full_format_recipes.csv")  # Assuming you have a CSV file with recipe data

# Example recipe titles corresponding to indices (you can load this from a file, here is a simple list)
recipe_titles = recipes_df['title'].tolist()

def recommend_similar(liked_recipe_indices, latent_embeddings, top_n=5):
    if not isinstance(latent_embeddings, np.ndarray):
        latent_embeddings = latent_embeddings.numpy()

    liked_recipe_indices = [idx for idx in liked_recipe_indices if 0 <= idx < len(latent_embeddings)]
    if not liked_recipe_indices:
        raise ValueError("All indices in liked_recipe_indices are out of bounds.")

    liked_embeddings = latent_embeddings[liked_recipe_indices]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(liked_embeddings, latent_embeddings)

    # Aggregate the scores for the liked recipes
    aggregated_scores = np.mean(similarity_scores, axis=0)

    # Get top_n recommended indices
    recommended_indices = np.argsort(-aggregated_scores)
    recommended_indices = [idx for idx in recommended_indices if idx not in liked_recipe_indices][:top_n]

    return recommended_indices

@app.route('/')
def home():
    # Automatically "like" some recipes (hardcoded indices for testing)
    liked_recipe_indices = [0, 5, 8]  # Example liked recipes
    top_n = 5  # Number of recommendations to return
    
    # Get recommendations
    recommended_indices = recommend_similar(liked_recipe_indices, latent_embeddings, top_n=top_n)
    
    # Fetch the recipe details based on recommended indices
    recommended_recipes_df = recipes_df.iloc[recommended_indices][['title', 'categories', 'ingredients']]
    
    # Extract the recipe titles and other details to pass to the template
    recommended_recipes = recommended_recipes_df.to_dict(orient='records')

    # Render the HTML page with recommended recipes
    return render_template('recommendation_page.html', recommended_recipes=recommended_recipes)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
