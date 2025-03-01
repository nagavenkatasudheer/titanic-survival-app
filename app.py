import streamlit as st
import pandas as pd
import pickle

# Load trained model (cached to prevent reloading every time)
@st.cache_resource
def load_model():
    with open('titanic_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Streamlit UI
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to predict survival probability.")

# User Inputs
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex", ['Male', 'Female'])
age = st.number_input("Age", min_value=1, max_value=100, value=30)
fare = st.number_input("Fare", min_value=0.0, value=50.0)
sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, value=0)
parch = st.number_input("Parents/Children Aboard", min_value=0, value=0)
embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'])

# Encode categorical variables
sex_encoded = 1 if sex == 'Female' else 0
embarked_encoded = {'C': 0, 'Q': 1, 'S': 2}[embarked]

# Make prediction
if st.button("Predict Survival"):
    input_data = pd.DataFrame([[pclass, sex_encoded, age, fare, sibsp, parch, embarked_encoded]],
                              columns=['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    st.markdown(f"### 🎯 Prediction: {'✅ Survived' if prediction == 1 else '❌ Did Not Survive'}")
    st.markdown(f"### 📊 Survival Probability: **{probability:.2f}**")

# Deployment Instructions
st.subheader("📌 Deployment Guide")
st.write("1. Save this script as `app.py` in your project folder.")
st.write("2. Ensure you have `titanic_model.pkl` (your trained model) in the same folder.")
st.write("3. Run the app locally using:")
st.code("streamlit run app.py", language="bash")
st.write("4. For online deployment, push your code to a GitHub repository and use Streamlit Community Cloud.")
st.write("[📖 Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app)")
