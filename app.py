import streamlit as st
from predictor import predict_prognosis


# Function to display the homepage
def show_homepage():
    st.title("Welcome to Health Predictor!")
    st.markdown("""
        This application is designed to help users assess their symptoms and provide potential diagnoses.
        It sources it's data from Symptom-Disease Prediction Dataset (SDPD), a comprehensive collection of 
        structured data linking symptoms to various diseases, meticulously curated to facilitate research and development 
        in predictive healthcare analytics.
        Using advanced machine learning algorithms and a fine-tuned BERT model, we aim to deliver 
        quick and reliable health predictions based on user-inputted symptoms.
    """)
    st.markdown("""
        ### Why This App?
        In today's fast-paced world, understanding your health is more important than ever. 
        Many individuals experience symptoms that can be concerning but may not have access to immediate medical advice.
        Additionally, resources that are readily available (e.g an internet search) are not often contexual to a patients'
        specific symptoms and circumstances.
        Our app aims to bridge that gap by providing a preliminary assessment of your symptoms and 
        guiding you on the next steps you should take to manage your health.
        It's important to note that none of the information that this app provides should be taken as official medical advice.
        Users should only use this tool as a possible assessment and not a definitive prognosis.
    """)
    
    # Add an image if available
    st.image("https://www.creativefabrica.com/wp-content/uploads/2020/01/05/Doctor-medical-check-up-for-healthcare-Graphics-1-40.jpg", caption="Picture Credits: Creative Fabrica", use_column_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col3:  # This column is the center column
        if st.button("START", key = "big_button"):
            st.session_state.page = "input"  # Navigate to the input page
            st.rerun()  # Force a rerun of the app to reflect the page change

    st.markdown(
        """
        <style>
        .stButton > button {
            font-size: 100px;
            padding: 20px 40px;
            background-color: #1f9990;
            color: white;
            border: black;
            border-radius: 5px;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.subheader("Resources")
    st.write("[Open Source Dataset](https://data.mendeley.com/datasets/dv5z3v2xyd/1) - Take a look at the dataset we used.")
    
    

# Function to display the input page
def show_input_page():
    st.title("Medical Symptom Checker")

    # Create a text area for user input
    user_input = st.text_area("Enter your symptoms. Be as detailed as possible to ensure a more accurate prognosis.", "e.g., Chest pain, shortness of breath")

    

    # Predict prognosis when the user clicks the button
    
    # Predict prognosis button
    if st.button("Predict Prognosis", key="predict-button"):
        if user_input:
            result, advice = predict_prognosis(user_input)
            st.success(f"Predicted Prognosis: {result}")
            st.write("### Health Advice:")
            st.write(advice)
        else:
            st.error("Please enter some symptoms.")
    

    
    # Back to home button
    if st.button("Back to Home", key="back-button"):
        st.session_state.page = "home"  # Navigate back to the homepage
        st.rerun()  # Force a rerun of the app to reflect the page change
    

# Main function to control the app flow
def main():
    # Initialize session state if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state.page = "home"  # Start on the homepage

    # Render the appropriate page based on the session state
    if st.session_state.page == "home":
        show_homepage()
    elif st.session_state.page == "input":
        show_input_page()

if __name__ == "__main__":
    main()