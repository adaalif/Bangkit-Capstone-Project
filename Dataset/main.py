from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the model and scaler
model = load_model("Model/autoencoder_model.h5")
with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load recipes dataset
recipes_df = pd.read_csv("recipes_with_final_embeddings.csv")

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Parse input JSON
        input_data = request.json
        liked_recipe_indices = input_data.get('liked_recipe_indices', [])
        top_n = input_data.get('top_n', 5)

        if not liked_recipe_indices:
            return jsonify({"error": "Please provide liked_recipe_indices"}), 400

        # Preprocess embeddings
        bert_embeddings = np.load("sbert_embeddings.npy")
        normalized_embeddings = scaler.transform(bert_embeddings)

        # Get latent embeddings from the model
        _, latent_embeddings = model(normalized_embeddings)
        latent_embeddings = latent_embeddings.numpy()

        # Validate indices
        liked_recipe_indices = [idx for idx in liked_recipe_indices if 0 <= idx < len(latent_embeddings)]
        if not liked_recipe_indices:
            return jsonify({"error": "Liked recipe indices are out of bounds"}), 400

        # Compute similarity
        liked_embeddings = latent_embeddings[liked_recipe_indices]
        similarity_scores = cosine_similarity(liked_embeddings, latent_embeddings)
        aggregated_scores = np.mean(similarity_scores, axis=0)

        # Recommend recipes
        recommended_indices = np.argsort(-aggregated_scores)
        recommended_indices = [idx for idx in recommended_indices if idx not in liked_recipe_indices][:top_n]

        recommended_recipes = recipes_df.iloc[recommended_indices][['title', 'categories', 'ingredients']].to_dict(orient="records")

        return jsonify({"recommended_recipes": recommended_recipes})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API is working fine!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
