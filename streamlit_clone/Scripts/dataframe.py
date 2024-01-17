import streamlit as st
import pandas as pd
import json

def app():
    st.header("Database: Saved data ")
    
    # Reading data from JSON file
    with open('data.json', 'r') as json_file:
        data = json.load(json_file)
    
    st.subheader("1. Press to view the existing data:card_index_dividers:")
    if not data:
        st.warning("No rows available for deletion")
    elif st.button("Press"):
        df = pd.DataFrame(data)
        df['Email ID'] = df['Email ID'].astype(str)
        st.write(df)
    
    st.subheader("2. Press to delete specific business card data by row number:")
    
    if not data:
        st.warning("No rows available for deletion")
    else:
        row_number = st.number_input("Enter the row number to delete:", min_value=1, max_value=len(data))

        if st.button("Delete"):
            if 1 <= row_number <= len(data):
                # Remove the specified row from the loaded JSON data
                del data[row_number - 1]
                # Save updated data back to JSON file
                with open('data.json', 'w') as json_file:
                    json.dump(data, json_file, indent=4)
                st.success(f"Row {row_number} deleted successfully")
            else:
                st.warning("Invalid row number")