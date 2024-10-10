import os
import streamlit as st
import requests
from PIL import Image
import extra_streamlit_components as stx
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.config import Config

# Set the page config first
st.set_page_config(page_title="Eye Disease Diagnosis", layout="wide")

# Ensure the upload folder exists
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)

API_URL = "http://localhost:5000"

# Load custom CSS
try:
    with open('C:/Users/Ashish/pythonProject/eye_disease_diagnsis/frontend/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Custom styles not found. Using default styles.")


def login(username, password):
    response = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        return response.json()
    return None


def signup(username, email, password):
    response = requests.post(f"{API_URL}/signup", json={"username": username, "email": email, "password": password})
    if response.status_code == 201:
        return response.json()
    return None


def predict(file, patient_id):
    files = {"file": file}
    data = {"patient_id": patient_id}

    try:
        response = requests.post(f"{API_URL}/predict", files=files, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
            return None
    except requests.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None


def get_patient_history(patient_id):
    response = requests.get(f"{API_URL}/patient_history/{patient_id}")
    if response.status_code == 200:
        return response.json()
    return []


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None


def main():
    if not st.session_state.logged_in:
        st.markdown('<h1 class="centered">Welcome to Easy Optha</h1>', unsafe_allow_html=True)

        tab_choice = stx.tab_bar(data=[
            stx.TabBarItemData(id="login", title="Login", description=""),
            stx.TabBarItemData(id="signup", title="Signup", description=""),
        ])

        if tab_choice == "login":
            st.markdown("<h2>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", key="login_button"):
                result = login(username, password)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result['user_id']
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            st.markdown("<h2>Sign Up</h2>", unsafe_allow_html=True)
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up", key="signup_button"):
                if signup(new_username, new_email, new_password):
                    st.success("Account created successfully. Please login.")
                else:
                    st.error("Signup failed. Username or email might already exist.")

    else:
        st.title("Easy Optha")

        # About Us section
        st.markdown('<div id="about-us">', unsafe_allow_html=True)
        st.markdown(""" 
            ### About Us
            **Easy Optha** is an innovative platform designed to assist eye specialists in diagnosing complex eye diseases such as cataract, glaucoma, and diabetic retinopathy using cutting-edge machine learning and image analysis techniques. 
            Our mission is to provide healthcare professionals with reliable, fast, and precise diagnostic tools, improving patient care and outcomes.

            With the ever-growing volume of medical data, our system streamlines the diagnostic process, allowing doctors to focus on what matters most – their patients. By leveraging advancements in artificial intelligence, Easy Optha brings accuracy and efficiency to the forefront of ophthalmology.

            We are committed to:
            - **Accuracy**: Ensuring that each diagnosis is backed by robust machine learning algorithms.
            - **Speed**: Providing quick, real-time results to facilitate timely interventions.
            - **Reliability**: A platform trusted by specialists for its consistency and precision.

            Join us in revolutionizing eye care and helping more patients see a brighter future.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        menu = ["Upload Scan", "Patient History", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Upload Scan":
            st.subheader("Upload New Scan")
            patient_id = st.text_input("Patient ID")
            uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

            if uploaded_files is not None:
                for uploaded_file in uploaded_files:
                    # Save the uploaded file to a designated path
                    uploaded_file_path = os.path.join(Config.UPLOAD_FOLDER, uploaded_file.name)

                    # Open and write the uploaded file
                    with open(uploaded_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Attempt to open the saved image for inspection
                    try:
                        with Image.open(uploaded_file_path) as img:
                            # Resize the image while maintaining aspect ratio
                            max_size = (400, 400)
                            img.thumbnail(max_size)
                            st.image(img, caption='Uploaded Image.', use_column_width=False)  # Preview image
                    except Exception as e:
                        st.error(f"Error opening saved image: {str(e)}")  # Show error in the Streamlit app

                if st.button('Predict'):
                    if patient_id:
                        with st.spinner('Processing...'):
                            for uploaded_file in uploaded_files:
                                try:
                                    result = predict(uploaded_file, patient_id)
                                    if result:
                                        st.success(f"Predicted disease for {uploaded_file.name}: {result['disease']}")
                                        st.write("Probabilities:")
                                        for disease, prob in result['probabilities'].items():
                                            st.progress(prob)
                                            st.write(f"{disease}: {prob:.2%}")
                                    else:
                                        st.error(f"Prediction failed for {uploaded_file.name}. Please try again.")
                                except Exception as e:
                                    st.error(f"An error occurred during prediction for {uploaded_file.name}: {str(e)}")
                    else:
                        st.warning("Please enter a Patient ID before predicting.")
        elif choice == "Patient History":
            st.subheader("Patient History")
            history_patient_id = st.text_input("Enter Patient ID to view history")
            if st.button("View History", key="history_button"):
                if history_patient_id:
                    with st.spinner('Fetching history...'):
                        history = get_patient_history(history_patient_id)
                    if history:
                        for item in history:
                            st.write(f"File: {item['filename']}")
                            st.write(f"Prediction: {item['prediction']['disease']}")
                            st.write("---")
                    else:
                        st.info("No history found for this patient.")
                else:
                    st.warning("Please enter a Patient ID to view history.")

        elif choice == "Logout":
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.rerun()


if __name__ == "__main__":
    main()
