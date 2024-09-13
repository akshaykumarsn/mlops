# app.py
import streamlit as st
from clustering import train_clustering, predict_cluster

st.title('Clustering with KMeans')

# Train the model
model = train_clustering()
st.write("Model trained successfully!")

# Get user input for inferencing
user_input = st.text_input('Enter data (comma-separated values):', '1.0, 2.0')
data = list(map(float, user_input.split(',')))

# Predict cluster
cluster = predict_cluster(model, data)
st.write(f'Predicted cluster: {cluster[0]}')
