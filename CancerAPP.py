import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('BreastCancer_rfModel.sav', 'rb') as file:
    model = pickle.load(file)

# Define feature ranges
feature_ranges = {
    "mean texture": (9.71, 39.28),
    "mean area": (143.5, 2501.0),
    "mean smoothness": (0.05263, 0.1634),
    "mean concave points": (0.0, 0.2012),
    "mean symmetry": (0.106, 0.304),
    "area error": (6.802, 542.2),
    "compactness error": (0.002252, 0.1354),
    "concavity error": (0.0, 0.396),
    "concave points error": (0.0, 0.05279),
    "worst texture": (12.02, 49.54),
    "worst area": (185.2, 4254.0),
    "worst smoothness": (0.07117, 0.2226),
    "worst compactness": (0.02729, 1.058),
    "worst concavity": (0.0, 1.252),
    "worst concave points": (0.0, 0.291),
    "worst symmetry": (0.1565, 0.6638),
    "worst fractal dimension": (0.05504, 0.2075),
}

# Set the app title and header
st.title("Breast Cancer Risk Prediction")
st.markdown(
    "Welcome to the Breast Cancer Risk Prediction App. This app uses a machine learning model to estimate "
    "whether a tumor is benign or malignant based on input data. Please enter the following information to get your prediction."
)

# Add a custom CSS for background and ribbon colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #fff5e7; /* Light Pink */
    }
    .css-10oheav {
    background: #ffffff; /* Beige Ribbon */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#source for the colours:
#https://www.colorhexa.com/ffebcd

# Create a two-column layout for input fields
col1, col2 = st.columns(2)


input_data = {}
for i, (feature, (min_val, max_val)) in enumerate(feature_ranges.items()):
    if i % 2 == 0:
        with col1:
            input_data[feature] = st.slider(
                f"{feature} ({min_val} - {max_val})", min_val, max_val, min_val + (max_val - min_val) / 2
            )
    else:
        with col2:
            input_data[feature] = st.slider(
                f"{feature} ({min_val} - {max_val})", min_val, max_val, min_val + (max_val - min_val) / 2
            )


# Add pictures in the sidebar
st.sidebar.image('ribbon.jpg', use_column_width=True)
st.sidebar.image('symptoms.jpg', use_column_width=True)
#st.sidebar.image('image3.jpg', use_column_width=True)

# Add a section for image sources
st.sidebar.markdown("Image Sources:")
st.sidebar.markdown("1. [Image 1 Source](https://www.thermofisher.com/blog/proteomics/breast-cancer-prognostic-biomarkers/)")
st.sidebar.markdown("2. [Image 2 Source](https://www.check4cancer.com/advice-and-awareness/breast-cancer)")
#st.sidebar.markdown("3. [Image 3 Source](https://source3.com)")

if st.button("Predict"):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])
    # Make a prediction
    prediction = model.predict(input_df)
    # Display the prediction result
    if prediction[0] == 0:
        st.error("Based on the input data, the tumor seems to be malignant.")
    else:
        st.success("Based on the input data, the tumor seems to be benign.")
        st.baloons()

# Add a footer with your name, affiliation, and a link to the source code
st.markdown(
    """
    ---
    Created by Fanny HERMENTIER *Student at IE University*
    Source code available on [GitHub](https://github.com/FannyHermentier)
    """,
    unsafe_allow_html=True,
)