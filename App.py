import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset into a pandas dataframe
df = pd.read_csv("E://PROJECT-DIRECTORY-TO-MAKE-PROJECTS//STREAMLIT-FOOD-RECOMMENDER//PROJECT1//IndianFoodDatasetCSV.csv")

# Create a CountVectorizer object to convert recipe names into vectors
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['TranslatedRecipeName'])

# Compute the cosine similarity matrix of the recipe names
cosine_sim = cosine_similarity(count_matrix)

# Define a function that takes a recipe name as input and returns the top 5 similar recipes
def get_similar_recipes(recipe_name):
    # Get the index of the recipe
    index = df[df['TranslatedRecipeName']==recipe_name].index[0]

    # Get the cosine similarity scores of the recipe
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the similar recipes by similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the top 5 similar recipes
    sim_scores = sim_scores[1:6]
    recipe_indices = [i[0] for i in sim_scores]

    # Return the top 5 similar recipes
    return df['TranslatedRecipeName'].iloc[recipe_indices]

# Define the Streamlit app
def app():
    # Set the app title
    st.title('Food Recommender By Sundar')
    
    # Get the recipe name from the user
    recipe_name =  st.selectbox('ENTER A RECIPES NAME:',
df['TranslatedRecipeName'].values)
    # Get the similar recipes
    if st.button('Get similar recipes'):
        similar_recipes = get_similar_recipes(recipe_name)
        
        # Display the similar recipes
        for recipe in similar_recipes:
            st.write(recipe)

# Run the Streamlit app
if __name__ == '__main__':
    app()
