import streamlit as st
import joblib

st.title("Customer Segmentation")

income = st.number_input("Annual Income", min_value=0)
score = st.number_input("Spending Score", min_value=0)

if st.button("Predict"):
    try:
        model = joblib.load('D:\CODE\CustomerSegmentation\Customer_segmentation_model.pkl')
        result = model.predict([[income, score]])
        cluster = result[0]
        cluster_messages = {
            0: "Customers with medium annual income and medium annual spend",
            1: "Customers with high annual income but high annual spend",
            2: "Customers with low annual income and high annual spend",
            3: "Customers with high annual income but low annual spend",
            4: "Customers with low annual income and low annual spend"
        }
        st.success(f"Cluster: {cluster}")
        st.info(cluster_messages.get(cluster, "Unknown cluster"))
    except Exception as e:
        st.error(f"Error: {e}")
