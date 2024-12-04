# panggil dataset 
from dataset import get_recipes_df
recipes_df = get_recipes_df()

def maintain(weight, height, age, gender, activity_level): # maintain untuk ideal
    # Gender
    if gender == 1:  # Male
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 2:  # Female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        return "Data Invalid"
    # Aktivitas
    activity_multipliers = {1: 1.2, 2: 1.375, 3: 1.55, 4: 1.725, 5: 1.9}
    tdee = bmr * activity_multipliers[activity_level]
    return round(tdee, 2)


def cutting(weight, height, age, gender, activity_level): # cutting untuk Overweight
    tdee = maintain(weight, height, age, gender, activity_level)
    cutting_calories = tdee * 0.8
    return round(cutting_calories, 2)


def bulking(weight, height, age, gender, activity_level): # Bulking untuk Underweight
    tdee = maintain(weight, height, age, gender, activity_level)
    bulking_calories = tdee * 1.15
    return round(bulking_calories, 2)


def calories(weight, height, age, gender, activity_level):
    # Menghitung BMI
    bmi = weight / (height / 100) ** 2 
    if bmi < 18.5:
        bmi_classification = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_classification = "Ideal"
    elif bmi >= 25:
        bmi_classification = "Overweight"
    else:
        bmi_classification = "Data Invalid"
    
    print(f"BMI Classification: {bmi_classification}")

    #Bulking
    if bmi_classification == "Underweight":
        bulk = bulking(weight, height, age, gender, activity_level)
        min = bulk / 3
        max = bulk / 3 + 50 # tambahan kalori, agar data resep bisa bervarasi
        protein = weight * 2 # Berat Badan dikali 1.5 gr protein
        filtered_recipes = recipes_df[
        recipes_df['calories'].apply(lambda x: min <= x <= max) & # Mencari Kalori
        recipes_df['protein'].apply(lambda x: x >= protein)       # Mencari Protein tinggi
        ]
        print("Bulking:", bulk , "calories/day")
        print(filtered_recipes[['title']]) ### Sesuaiin sama yang mau ditampilin

    #Maintain
    elif bmi_classification == "Ideal":
        mt = maintain(weight, height, age, gender, activity_level)
        min = mt / 3 - 50
        max = mt / 3 + 50
        filtered_recipes = recipes_df[
        recipes_df['calories'].apply(lambda x: min <= x <= max) 
        ]
        print("Maintain:", mt, "calories/day")
        print(filtered_recipes[['title']]) ### Sesuaiin sama yang mau ditampilin

    #Cutting
    elif bmi_classification == "Overweight":
        cut = cutting(weight, height, age, gender, activity_level)
        min = cut / 3 - 50
        max = cut / 3
        protein = weight * 1.5 # Berat Badan dikali 1.5 gr protein
        filtered_recipes = recipes_df[
        recipes_df['calories'].apply(lambda x: min <= x <= max) & 
        recipes_df['protein'].apply(lambda x: x >= protein)
        ]
        print("Cutting:", cut, "calories/day")
        print(filtered_recipes[['title']]) ### Sesuaiin sama yang mau ditampilin

