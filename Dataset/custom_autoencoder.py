# model.py
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics.pairwise import cosine_similarity

class CustomAutoencoder(tf.keras.Model):
    def __init__(self, input_dim=768, latent_dim=512):
        super(CustomAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.input_noise = layers.GaussianNoise(0.05)  
        self.encoder_dense1 = layers.Dense(
            latent_dim, activation='relu', 
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.encoder_dense2 = layers.Dense(
            256, activation='relu', 
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.encoder_latent = layers.Dense(
            latent_dim, activation='relu',
            kernel_regularizer=regularizers.l1(1e-5)  
        )
        self.latent_noise = layers.GaussianNoise(0.05)
        self.latent_normalization = layers.LayerNormalization()

        # Decoder
        self.decoder_dense1 = layers.Dense(
            256, activation='relu', 
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.decoder_dense2 = layers.Dense(
            512, activation='relu', 
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.decoder_output = layers.Dense(input_dim, activation='sigmoid')

    def call(self, inputs):
        # Encoder
        x = self.input_noise(inputs)  
        x = self.encoder_dense1(x)
        skip = x  
        x = self.encoder_dense2(x)
        latent = self.encoder_latent(x)
        latent = self.latent_noise(latent)  
        latent = self.latent_normalization(latent)  
        latent = layers.add([latent, skip])  

        # Decoder
        x = self.decoder_dense1(latent)
        x = self.decoder_dense2(x)
        reconstruction = self.decoder_output(x)
        return reconstruction, latent

    def get_config(self):
        config = super(CustomAutoencoder, self).get_config()
        config.update({
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Extract only the relevant parameters for the CustomAutoencoder
        input_dim = config.pop('input_dim', 768)  # Default to 768 if not found
        latent_dim = config.pop('latent_dim', 512)  # Default to 512 if not found
        return cls(input_dim=input_dim, latent_dim=latent_dim)

class RecipeRecommender:
    def __init__(self, model_path, scaler_path, recipes_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path, custom_objects={'CustomAutoencoder': CustomAutoencoder})
        
        # Load the scaler
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        # Load the recipes DataFrame
        self.recipes_df = pd.read_csv(recipes_path)
        
    def recommend(self, liked_recipe_indices, top_n=5):
        # Get latent embeddings from the model
        normalized_embeddings = self.scaler.transform(self.recipes_df['embeddings'].values.tolist())
        latent_embeddings = self.model(normalized_embeddings)[1].numpy()

        # Calculate cosine similarity
        liked_embeddings = latent_embeddings[liked_recipe_indices]
        similarity_scores = cosine_similarity(liked_embeddings, latent_embeddings)
        aggregated_scores = np.mean(similarity_scores, axis=0)

        # Get recommended indices
        recommended_indices = np.argsort(-aggregated_scores)
        recommended_indices = [idx for idx in recommended_indices if idx not in liked_recipe_indices][:top_n]

        return self.recipes_df.iloc[recommended_indices][['title', 'categories', 'ingredients']]

# Example usage:
# recommender = RecipeRecommender('custom_autoencoder_model.h5', 'model/scaler.pkl', 'full_format_recipes.csv')
# recommendations = recommender.recommend([76