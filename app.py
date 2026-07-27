import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from groq_ai import ask_ai
# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Student Performance Prediction AI",
    page_icon="🎓",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------

model = joblib.load("decision_tree.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("StudentsPerformance.csv")

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("🎓 Student Performance AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "EDA",
        "Prediction",
        "AI Assistant"
    ]
)

# ----------------------------
# HOME PAGE
# ----------------------------

if page == "Home":

    st.title("🎓 Student Performance Prediction")

    st.markdown("""
This application predicts **Math Score**
using a **Decision Tree Regressor**.

### Features

- Dataset Preview
- Exploratory Data Analysis
- Student Score Prediction
- AI Assistant
- Streamlit Deployment
""")

    st.success("Model Loaded Successfully!")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Model", "Decision Tree")

# ----------------------------
# DATASET PAGE
# ----------------------------

elif page == "Dataset":

    st.title("📄 StudentsPerformance Dataset")

    st.subheader("First Five Rows")

    st.dataframe(df.head())

    st.subheader("Dataset Shape")

    st.write(df.shape)

    st.subheader("Column Names")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

    st.subheader("Statistics")

    st.dataframe(df.describe())

# ----------------------------
# EDA PAGE
# ----------------------------

elif page == "EDA":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Math Score Distribution")

    fig = px.histogram(
        df,
        x="math score",
        nbins=20
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Reading Score Distribution")

    fig = px.histogram(
        df,
        x="reading score",
        nbins=20
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Writing Score Distribution")

    fig = px.histogram(
        df,
        x="writing score",
        nbins=20
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender Count")

    fig = px.bar(
        df["gender"].value_counts().reset_index(),
        x="gender",
        y="count"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Lunch Type")

    fig = px.pie(
        df,
        names="lunch"
    )

    st.plotly_chart(fig, use_container_width=True)
    # ----------------------------
# PREDICTION PAGE
# ----------------------------

elif page == "Prediction":

    st.title("🎯 Student Performance Prediction")

    st.write("Enter the student's information below.")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["female", "male"]
        )

        race = st.selectbox(
            "Race/Ethnicity",
            [
                "group A",
                "group B",
                "group C",
                "group D",
                "group E"
            ]
        )

        parental = st.selectbox(
            "Parental Level of Education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
        )

        lunch = st.selectbox(
            "Lunch",
            [
                "standard",
                "free/reduced"
            ]
        )

    with col2:

        preparation = st.selectbox(
            "Test Preparation Course",
            [
                "none",
                "completed"
            ]
        )

        reading = st.slider(
            "Reading Score",
            0,
            100,
            70
        )

        writing = st.slider(
            "Writing Score",
            0,
            100,
            70
        )

    if st.button("Predict Math Score"):

        input_df = pd.DataFrame({

            "gender":[gender],

            "race/ethnicity":[race],

            "parental level of education":[parental],

            "lunch":[lunch],

            "test preparation course":[preparation],

            "reading score":[reading],

            "writing score":[writing]

        })

        processed = preprocessor.transform(input_df)

        prediction = model.predict(processed)

        st.success(
            f"🎉 Predicted Math Score : {prediction[0]:.2f}"
        )

        st.subheader("Entered Student Information")

        st.dataframe(input_df)

        st.subheader("Prediction Summary")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Reading Score",
            reading
        )

        c2.metric(
            "Writing Score",
            writing
        )

        c3.metric(
            "Predicted Math",
            round(float(prediction[0]),2)
        )

        chart_df = pd.DataFrame({

            "Subject":[
                "Reading",
                "Writing",
                "Predicted Math"
            ],

            "Score":[
                reading,
                writing,
                prediction[0]
            ]

        })

        fig = px.bar(

            chart_df,

            x="Subject",

            y="Score",

            text="Score"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
        # ----------------------------
# AI ASSISTANT PAGE
# ----------------------------

elif page == "AI Assistant":

    st.title("🤖 Student Performance AI Assistant")

    st.write(
        "Ask any question related to the project, Decision Tree, EDA, or Machine Learning."
    )

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask AI"):

        if question.strip() == "":

            st.warning("Please enter a question.")

        else:

            with st.spinner("Thinking..."):

                try:

                    answer = ask_ai(question)

                    st.success("Answer")

                    st.write(answer)

                except Exception as e:

                    st.error("Error")

                    st.code(str(e))

# ----------------------------
# FOOTER
# ----------------------------

st.markdown("---")

st.markdown(
    """
### 📚 Project Information

**Project:** Student Performance Prediction AI

**Machine Learning Model:** Decision Tree Regressor

**Dataset:** StudentsPerformance.csv

**Libraries Used**
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq Llama 3

---

👨‍💻 Developed using Python, Machine Learning, LangChain, and Streamlit.

"""
)
