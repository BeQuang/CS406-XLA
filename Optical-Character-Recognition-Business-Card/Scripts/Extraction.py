import streamlit as st
import easyocr
import matplotlib.pyplot as plt
from IPython.display import Image
# import mysql.connector
# mydb=mysql.connector.Connect(host="localhost",port=3306, user="root",password="pass",database="BCData")
# mycursor=mydb.cursor()

plt.rcParams['figure.figsize'] = 8,16
reader = easyocr.Reader(['en'],gpu=False)
def extract_data(image):
    data=reader.readtext(image)
    name=""
    des=""
    phone_number=""
    email=""
    website=""
    remaining=[]
    for i in range(0,len(data)):
        if i==0:
            name=data[i][1]
        elif i==1:
            des=data[i][1]
        elif "-" in data[i][1] or "+" in data[i][1]:
            phone_number=data[i][1]
        elif "@" in data[i][1] and ".com" in data[i][1]:
            email=data[i][1]
        elif "www." in data[i][1] or ".com" in data[i][1]:
            website=data[i][1]
        else:
            remaining.append(data[i][1])
    address=""
    company_name=""
    for i in remaining:
        count=0
        for j in i:
            if j in "1234567890" or j in ";:,.":
                count+=1
        if count>0:
            address=address+i+" "
        elif i.lower()=="www":
            website=i+"."+website
        else:
            company_name=company_name+i+' '
    website1=website.replace(" ",".")
    st.subheader("Extracted information from the uploaded image.")
    st.write(f"Employee Name: {name}")
    st.write(f"Designation: {des}")
    st.write(f"Company Name: {company_name}")
    st.write(f"Contact Number: {phone_number}")
    st.write(f"Email ID: {email}")
    st.write(f"Website: {website1}")
    st.write(f"Address: {address}")

    st.subheader("Database uploadation!!")
    
    option=st.radio("Upload the extracted data on the database server:",(" ","Yes","No"))
    if option=="Yes":
        # mycursor.execute(f"INSERT INTO business_card_info VALUES ('{name}','{des}','{company_name}','{phone_number}','{email}','{website}','{address}','{image}')")
        # mydb.commit()
        st.success("Uploaded Successfully!",icon='✅')
        st.balloons()
    elif option=="No":
        st.write("Thank you!")

    

    