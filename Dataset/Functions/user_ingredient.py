# panggil dataset 
from dataset import get_recipes_df
recipes_df = get_recipes_df()

ingredients = { # Button untuk milih bahan2 bisa disesuai ke index dictionary
    # Protein
    1: 'Chicken', 2: 'Fish', 3: 'Beef', 4: 'Lamb', 5: 'Egg', 6: 'Tofu', 7: 'Shrimp', 8: 'Squid', 9: 'Clam', 10: 'Crab', 11: 'Sausage', 12: 'Pork',
    # Vegetables
    13: 'Spinach', 14: 'Cabbage', 15: 'Eggplant', 16: 'Tomato', 17: 'Cauliflower', 18: 'Lettuce', 19: 'Bok Choy', 20: 'Bean', 21: 'Carrot', 22: 'Broccoli', 23: 'Kale', 24: 'Celery',
    # Carbs
    25: 'Rice', 26: 'Bread', 27: 'Potato', 28: 'Corn', 29: 'Noodle', 30: 'Pasta',
    # Fruits
    31: 'Apple', 32: 'Banana', 33: 'Orange', 34: 'Mango', 35: 'Grapes', 36: 'Pineapple', 37: 'Watermelon', 38: 'Strawberry', 39: 'Lemon', 40: 'Avocado', 41: 'Coconut', 42: 'Durian', 43: 'Guava', 44: 'Berry',
    # Condiments
    45: 'Milk', 46: 'Cheese', 47: 'Soy', 48: 'Tamarind'
} 

def recipe_ingredient():
   # list kosong buat nyimpen bahan pilihan user
    selected_ingredients = []

    # input user 
    for i in range(3): # maksimal 3x
        ingredient_input = input(f"Enter ingredient {i+1} (type 'x' to finish): ") # [1] bisa ganti ke action button kyk "Submit Ingredient" 
        
        if ingredient_input.lower() == 'x' and len(selected_ingredients) > 0: # [2] ganti disini juga
            break
        
        try:
            ingredient_index = int(ingredient_input)
            if 1 <= ingredient_index <= len(ingredients):
                selected_ingredients.append(ingredients[ingredient_index])
        except ValueError:
            print("Choose ingredient first !") # kasih alert kalo user belum milih bahan tapi udah submit

    # bahan-bahan yang udah masuk ke list
    print("Selected ingredients:")
    for ingredient in selected_ingredients:
        print(ingredient)

    # filter bahan yang ada di dataset (resep)
    filtered_recipes = recipes_df[recipes_df['categories'].apply(lambda x: all(ingredient.lower() in x.lower() for ingredient in selected_ingredients))]

    print(filtered_recipes[['title']]) ### Sesuaiin sama yang mau ditampilin
