from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
from model import CustomAutoencoder
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model with custom layers
try:
    loaded_model = tf.keras.models.load_model('custom_autoencoder_model.keras', custom_objects={'CustomAutoencoder': CustomAutoencoder})
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Load latent embeddings
latent_embeddings = np.load("latent_embeddings.npy")  # Replace with your latent embeddings file

# Load the dataset
recipes_df = pd.read_csv("full_format_recipes.csv")  # Replace with your actual dataset
recipe_titles = recipes_df['title'].tolist()

# Utility function to recommend recipes based on latent similarity
def recommend_similar(liked_recipe_indices, filtered_latent_embeddings, top_n=10, candidate_pool=30):
    if not isinstance(filtered_latent_embeddings, np.ndarray):
        filtered_latent_embeddings = filtered_latent_embeddings.numpy()

    # Ensure indices are within bounds
    liked_recipe_indices = [idx for idx in liked_recipe_indices if 0 <= idx < len(filtered_latent_embeddings)]
    if not liked_recipe_indices:
        raise ValueError("All indices in liked_recipe_indices are out of bounds.")

    liked_embeddings = filtered_latent_embeddings[liked_recipe_indices]

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(liked_embeddings, filtered_latent_embeddings)

    # Aggregate the scores for the liked recipes
    aggregated_scores = np.mean(similarity_scores, axis=0)

    # Take the most similar candidate_pool
    candidate_indices = np.argsort(-aggregated_scores)
    candidate_indices = [idx for idx in candidate_indices if idx not in liked_recipe_indices][:candidate_pool]

    # Choose random top_n from candidate
    if len(candidate_indices) < top_n:
        top_n = len(candidate_indices)
    
    recommended_indices = np.random.choice(candidate_indices, size=top_n, replace=False)

    return recommended_indices.tolist()

# Helper function to filter the dataset based on calorie constraints
def filter_dataset(target_calories, tolerance=200):
    """
    Filters the recipes dataset to include recipes within the specified calorie range.
    """
    min_calories = target_calories - tolerance
    max_calories = target_calories + tolerance

    filtered_df = recipes_df[
        (recipes_df['calories'] >= min_calories) & 
        (recipes_df['calories'] <= max_calories)
    ]
    return filtered_df

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user inputs
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        activity_level = int(request.form['activity_level'])

        # Calculate BMI
        bmi = weight / (height / 100) ** 2
        if bmi < 18.5:
            bmi_status = "Underweight"
            calorie_target = maintain(weight, height, age, gender, activity_level) * 1.15
        elif 18.5 <= bmi < 24.9:
            bmi_status = "Ideal Weight"
            calorie_target = maintain(weight, height, age, gender, activity_level)
        else:
            bmi_status = "Overweight"
            calorie_target = maintain(weight, height, age, gender, activity_level) * 0.8

        # Calculate target calories per meal
        target_calories_per_meal = calorie_target / 3

        # Step 1: Filter the dataset based on calorie constraints (±200 calories of target)
        filtered_df = filter_dataset(target_calories_per_meal, tolerance=200)

        # Update latent embeddings for the filtered dataset
        filtered_indices = filtered_df.index.tolist()
        filtered_latent_embeddings = latent_embeddings[filtered_indices]

        # Step 2: Generate recommendations based on the filtered dataset
        liked_recipe_indices = [0, 1, 2]  # Placeholder for liked recipes (adjust based on user interaction)
        top_n = 10  # Number of recommendations to return

        # Generate recommendations
        filtered_recommendation_indices = recommend_similar(
            liked_recipe_indices, filtered_latent_embeddings, top_n=top_n
        )

        # Map filtered indices back to the original dataset
        final_recommendations = filtered_df.iloc[filtered_recommendation_indices]

        # Step 3: Prepare data for rendering
        final_recommendations_data = final_recommendations.to_dict(orient='records')

        # Render the recommendations page
        return render_template(
            'recommendation_page.html',
            recommended_recipes=final_recommendations_data,
            target_calories_per_meal=round(target_calories_per_meal, 2),  # Pass target calories to template
            bmi_status=bmi_status  # Pass BMI status to template
        )

    # Render the input form on GET requests
    return render_template('home.html')


# Helper function to calculate TDEE
def maintain(weight, height, age, gender, activity_level):
    if gender == 1:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 2:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        return "Invalid data"

    activity_multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    tdee = bmr * activity_multipliers[activity_level]
    return round(tdee, 2)

# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
