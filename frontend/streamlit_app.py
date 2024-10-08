import streamlit as st
import requests
from PIL import Image
import io
import extra_streamlit_components as stx

# Set the page config first
st.set_page_config(page_title="Eye Disease Diagnosis", layout="wide")

API_URL = "http://localhost:5000"

# Load custom CSS
try:
    # Adjust the path according to where your styles.css is located
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
    response = requests.post(f"{API_URL}/predict", files=files, data=data)
    if response.status_code == 200:
        return response.json()
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
        st.markdown('<div class="centered">', unsafe_allow_html=True)

        tab_choice = stx.tab_bar(data=[
            stx.TabBarItemData(id="login", title="Login", description=""),
            stx.TabBarItemData(id="signup", title="Signup", description=""),
        ])

        if tab_choice == "login":
            st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.button("Login", key="login_button", help="Click to log in"):
                result = login(username, password)
                if result:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result['user_id']
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        else:
            st.markdown("<h2 style='text-align: center;'>Sign Up</h2>", unsafe_allow_html=True)
            new_username = st.text_input("New Username")
            new_email = st.text_input("Email")
            new_password = st.text_input("New Password", type="password")
            if st.button("Sign Up", key="signup_button", help="Click to create an account"):
                if signup(new_username, new_email, new_password):
                    st.success("Account created successfully. Please login.")
                else:
                    st.error("Signup failed. Username or email might already exist.")

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.title("Eye Disease Diagnosis Dashboard")

        menu = ["Upload Scan", "Patient History", "Logout"]
        choice = st.sidebar.selectbox("Menu", menu)

        if choice == "Upload Scan":
            st.subheader("Upload New Scan")
            patient_id = st.text_input("Patient ID")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                if st.button('Predict', key="predict_button", help="Click to predict disease"):
                    if patient_id:
                        result = predict(uploaded_file, patient_id)
                        if result:
                            st.success(f"Predicted disease: {result['disease']}")
                            st.info(f"Confidence: {result['confidence']:.2f}")
                        else:
                            st.error("Error in prediction. Please try again.")
                    else:
                        st.warning("Please enter a Patient ID before predicting.")

        elif choice == "Patient History":
            st.subheader("Patient History")
            history_patient_id = st.text_input("Enter Patient ID to view history")
            if st.button("View History", key="history_button", help="Click to view patient history"):
                if history_patient_id:
                    history = get_patient_history(history_patient_id)
                    if history:
                        for item in history:
                            st.write(f"File: {item['filename']}")
                            st.write(
                                f"Prediction: {item['prediction']['disease']} (Confidence: {item['prediction']['confidence']:.2f})")
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
