import streamlit as st
import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import fuzz

def calculate_similarity(search_input, df):
    # Combine all text columns into a single column for similarity calculation
    text_columns = ['Employee Name', 'Designation', 'Company Name', 'Contact Number', 'Email ID', 'Website', 'Address']
    df['combined_text'] = df[text_columns].astype(str).agg(' '.join, axis=1)

    # Create TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_text'])

    # Calculate cosine similarity
    search_vector = tfidf_vectorizer.transform([search_input])
    cosine_similarities = linear_kernel(search_vector, tfidf_matrix).flatten()

    # Calculate fuzzy similarity using fuzzywuzzy
    fuzzy_scores = df['combined_text'].apply(lambda x: fuzz.token_sort_ratio(search_input, x))
    
    # Combine cosine similarities and fuzzy scores (you can adjust the weights)
    combined_scores = 0.7 * cosine_similarities + 0.3 * (fuzzy_scores / 100.0)

    # Sort the results based on combined scores
    df['similarity'] = combined_scores
    df = df.sort_values(by='similarity', ascending=False)

    return df[['Employee Name', 'Designation', 'Company Name', 'Contact Number', 'Email ID', 'Website', 'Address', 'similarity']]


def app():
    st.header("Database: Saved data ")

    # Get the absolute path to the 'data.json' file
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(script_directory, 'data.json')

    # Check if the file exists
    if not os.path.exists(data_file_path):
        st.warning(f"File '{data_file_path}' not found.")
        return

    # Reading data from JSON file
    try:
        with open(data_file_path, 'r') as json_file:
            data = json.load(json_file)
    except Exception as e:
        st.error(f"Error reading data from '{data_file_path}': {str(e)}")
        return

    df = pd.DataFrame(data)
    df['Email ID'] = df['Email ID'].astype(str)

    st.subheader("1. Press to view the existing data:card_index_dividers:")
    if not data:
        st.warning("No rows available for deletion")
    elif st.button("Press"):
        st.write(df)

    st.subheader("2. Press to delete specific business card data by row number:")
    if not data:
        st.warning("No rows available for deletion")
    else:
        row_number = st.number_input("Enter the row number to delete:", min_value=0, max_value=len(data))

        if st.button("Delete"):
            if 0 <= row_number <= len(data):
                # Remove the specified row from the loaded JSON data  
                del data[row_number]
                # Save updated data back to JSON file
                try:
                    with open(data_file_path, 'w') as json_file:
                        json.dump(data, json_file, indent=4)
                    st.success(f"Row {row_number} deleted successfully")
                except Exception as e:
                    st.error(f"Error saving data to '{data_file_path}': {str(e)}")
            else:
                st.warning("Invalid row number")

    st.subheader("3. Search for specific rows:")
    search_input = st.text_input("Enter search information:")

    if st.button("Search"):
        if search_input:
            # Filter rows based on the search input
            filtered_df = calculate_similarity(search_input, df)
            
            # Display the top K results (e.g., top 5)
            st.write(filtered_df.head(5))
        else:
            st.warning("Please enter a search query")


# Call the app function if running as a script
if __name__ == "main":
    app()