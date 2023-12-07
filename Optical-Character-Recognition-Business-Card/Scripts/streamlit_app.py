# app.py
import streamlit as st
import home_page
# import database

PAGES = {
    "Home": home_page,
    # "Database": database
}

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", list(PAGES.keys()))

    PAGES[page].app()

if __name__ == "__main__":
    main()
