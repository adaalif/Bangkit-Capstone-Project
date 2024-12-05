import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, GaussianNoise
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from umap import UMAP

# Load dataset
recipes_df = pd.read_csv("full_format_recipes.csv")
bert_embeddings = np.load("sbert_embeddings.npy")
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(bert_embeddings)
input_dim = bert_embeddings.shape[1]  

class CustomAutoencoder(tf.keras.Model):
    def __init__(self, input_dim = normalized_embeddings.shape[1], latent_dim = 512, **kwargs):
        super(CustomAutoencoder, self).__init__(**kwargs)
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
        return cls(**config)