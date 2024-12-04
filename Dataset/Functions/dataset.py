## Sesuaikan sama lokasi dataset cloud

# Baca Data dari CSV
import pandas as pd

recipes_df = pd.read_csv("recipes_with_final_embeddings.csv")
pd.set_option('display.max_colwidth', None)

def get_recipes_df():
    return recipes_df

