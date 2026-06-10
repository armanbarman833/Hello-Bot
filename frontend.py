import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# Load model
model = joblib.load("model.pkl")

# Page Config
st.set_page_config(
    page_title="AI Marks Predictor",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
}

.big-font {
    font-size:40px !important;
    font-weight: bold;
    color: cyan;
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<p class="big-font">🤖 AI Student Performance Predictor</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# Sidebar
st.sidebar.title("📌 Project Information")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("""
### Technologies Used
- Python
- Scikit-learn
- Streamlit
- Matplotlib
- Pandas
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "This AI predicts student marks based on study hours."
)

# Layout
col1, col2 = st.columns(2)

# Left Column
with col1:

    st.subheader("📘 Student Input")

    hours = st.slider(
        "Study Hours",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.5
    )

    study_level = st.selectbox(
        "Study Consistency",
        [
            "Low",
            "Medium",
            "High"
        ]
    )

    sleep = st.slider(
        "Sleep Hours",
        0,
        10,
        7
    )

    motivation = st.select_slider(
        "Motivation Level",
        options=[
            "Poor",
            "Average",
            "Good",
            "Excellent"
        ]
    )

# Right Column
with col2:

    st.subheader("📊 AI Prediction")

    if st.button("🚀 Predict Now"):

        # Loading animation
        with st.spinner("AI is analyzing data..."):

            time.sleep(2)

        prediction = model.predict([[hours]])

        predicted_marks = round(prediction[0], 2)

        # Success
        st.success(
            f"🎯 Predicted Marks: {predicted_marks}"
        )

        # Progress bar
        st.progress(int(predicted_marks))

        # Metrics
        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Study Hours",
            f"{hours}"
        )

        m2.metric(
            "Expected Marks",
            f"{predicted_marks}"
        )

        m3.metric(
            "Sleep",
            f"{sleep} hrs"
        )

        # Performance Message
        if predicted_marks >= 80:
            st.balloons()
            st.success("🔥 Excellent Performance Expected!")

        elif predicted_marks >= 60:
            st.info("👍 Good Performance")

        else:
            st.warning("⚠️ Need More Practice")

        # Graph
        st.markdown("## 📈 Prediction Visualization")

        x = np.arange(0, 11, 1)

        y = model.predict(x.reshape(-1,1))

        fig, ax = plt.subplots(figsize=(8,5))

        ax.plot(x, y)

        ax.scatter(hours, predicted_marks)

        ax.set_xlabel("Study Hours")
        ax.set_ylabel("Marks")

        ax.set_title("AI Prediction Graph")

        st.pyplot(fig)

        # Data Table
        st.markdown("## 📋 Sample Dataset")

        sample_data = pd.DataFrame({
            "Hours": [1,2,3,4,5,6,7,8,9,10],
            "Marks": [20,25,35,40,50,60,70,80,90,95]
        })

        st.dataframe(sample_data)

# Footer
st.markdown("---")

st.caption(
    "🚀 Built with Machine Learning + AI Dashboard using Streamlit"
)