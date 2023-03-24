import streamlit as st
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Load data
@st.cache
def load_data():
    data = pd.read_excel('C:/Users/HIMANGI/Downloads/daily_offers/daily_offers.xlsx')
    return data

df = load_data()

# Encode categorical features
le = LabelEncoder()
df['status_encoded'] = le.fit_transform(df['status'])
df['item_type_encoded'] = le.fit_transform(df['item type'])

# Train Lasso regression model
alpha = st.sidebar.slider("alpha", min_value=0.0, max_value=1.0, step=0.01, value=0.1)
model = Lasso(alpha=alpha, precompute=True, positive=True, selection='random', random_state=42)

# Create text input boxes for independent variables
application = st.text_input("Application")
thickness = st.text_input("Thickness")
width = st.text_input("Width")
status_encoded = st.text_input("Status Encoded")
item_type_encoded = st.text_input("Item Type Encoded")

# Make predictions based on user input
if st.button("Predict Selling Price"):
    X = pd.DataFrame({'application': [float(application)],
                      'thickness': [float(thickness)],
                      'width': [float(width)],
                      'status_encoded': [float(status_encoded)],
                      'item_type_encoded': [float(item_type_encoded)]})
    y_pred = model.predict(X)
    st.write("Predicted selling price:", y_pred[0])

# Display results
st.header("Lasso Regression Model")
st.write("R-squared score:", model.score(X, y))

# Create a slider to adjust the regularization parameter
st.sidebar.header("Lasso Regression Parameters")
st.sidebar.write("Adjust the regularization parameter alpha.")
