import streamlit as st
import mysql.connector
import pandas as pd

mydb=mysql.connector.Connect(host="localhost",port=3306, user="root",password="pass",database="BCData")
mycursor=mydb.cursor()

def app():
    st.sidebar.success("Database: Saved data ")
    st.header(" Saved data management 🎈")
    st.subheader("Press to view the existing data:card_index_dividers:")
    if st.button("Press"):
        mycursor.execute("select * from business_card_info")
        data=[]
        for i in mycursor:
            data.append(i)
        df=pd.DataFrame(data,columns=["Employee_name","Designation","Company Name","Contact_number","Email_id","Website","Address","card"])
        st.write(df)
    
    st.subheader("Press to delete specific business card data:small_red_triangle_down:")
    a=st.text_input("Enter the employee name that is to be deleted:")
    if st.button("Delete"):
        if len(a)>0:
            mycursor.execute(f"delete from business_card_info where name='{a}'")
            mydb.commit()
            st.success("Row deleted successfully")
        else:
            st.warning("No row affected")            