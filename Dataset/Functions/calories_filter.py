from dataset import get_recipes_df
recipes_df = get_recipes_df()

# Dictionary rentang kalori
calorie_content = {
    1: (0, 500), # rentang <500
    2: (501, 1000), # rentang 501-1000
    3: (1001, 1500) # rentang 1001-1500
}

# Fungsi filter kalori 
def recipe_calories_filter():

    # Variabel untuk menyimpan nilai rentang kalori
    selected_calories_range = None

    # Input user
    calorie_input = input("Enter your calorie choice (1-3 or type apply to confirm): ").strip()

    if calorie_input.lower() == 'apply': # Opsi Konfirmasi
        if not selected_calories_range: # Jika belum ada pilihan
            print("You must choose calories range first!")
            return
    else:
        try:
            calorie_choice = int(calorie_input)
            if 1 <= calorie_choice <= len(calorie_content): # Validasi input
                selected_calories_range = calorie_content[calorie_choice]
            else:
                raise ValueError("Invalid choice!")
        except ValueError:
            print("Choice a valid categorie range before submitting!")
            return
        
    # Menampilkan rentang kalori yang dipilih
    if selected_calories_range:
        print(f"Selected calorie range: {selected_calories_range[0]} - {selected_calories_range[1]} Calories")

    # Filter resep berdasarkan rentang kalori yang dipilih
    calorie_min, calorie_max = selected_calories_range
    filtered_recipes = recipes_df[
        {recipes_df['calories'] >= calorie_min} &
        {recipes_df['calories'] <= calorie_max}
    ]

    # Pilih 20 resep secara acak
    filtered_recipes = filtered_recipes.sample(n=min(20, len(filtered_recipes)))

    # Menampilkan hasil filter
    if filtered_recipes.empty():
        print("\nNo recipes found")
    else:
        print("\nRecommended Recipes:")
        print(filtered_recipes[['title']])