"""
app.py  —  Streamlit web application for Student Performance Predictor
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  CUSTOM CSS  —  clean, academic feel
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio label { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 { color: white !important; }

    /* Main background */
    .main { background-color: #f7f9fc; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border-left: 4px solid #2c5364;
    }

    /* Prediction result box */
    .result-pass {
        background: linear-gradient(135deg, #d4edda, #c3e6cb);
        border-left: 6px solid #28a745;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-size: 1.1rem;
    }
    .result-fail {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left: 6px solid #dc3545;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-size: 1.1rem;
    }
    .result-score {
        background: linear-gradient(135deg, #cce5ff, #b8daff);
        border-left: 6px solid #004085;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        font-size: 1.1rem;
    }

    /* Section header */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2c5364;
        border-bottom: 3px solid #2c5364;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }

    /* Tip box */
    .tip-box {
        background: #fffde7;
        border-left: 4px solid #f9a825;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/student_data.csv")

@st.cache_resource
def load_models():
    score_model    = joblib.load("models/score_model.pkl")
    pass_fail_model= joblib.load("models/pass_fail_model.pkl")
    return score_model, pass_fail_model

@st.cache_data
def load_metrics():
    with open("models/metrics.json") as f:
        return json.load(f)

def encode_input(study, attend, prev, sleep, internet, par_edu, extra):
    """Turn user inputs into the feature vector the models expect."""
    edu_order = {"None":0,"Primary":1,"Secondary":2,"Bachelor":3,"Master":4}
    return np.array([[
        study,
        attend,
        prev,
        sleep,
        1 if internet == "Yes" else 0,
        edu_order[par_edu],
        1 if extra == "Yes" else 0,
    ]])


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Student Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate to",
        ["🏠 Home",
         "📋 Dataset Preview",
         "📊 Visualizations",
         "🔮 Make a Prediction",
         "📈 Model Performance",
         "ℹ️ About Project"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#aaa'>Built with Python · Scikit-learn · Streamlit</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 1 — HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown('<p class="section-header">🎓 Student Performance Predictor</p>',
                unsafe_allow_html=True)
    st.markdown("""
    Welcome! This app uses **Machine Learning** to predict how well a student might perform
    based on simple factors like study hours, attendance, and previous scores.

    ---
    ### 🚀 What can this app do?
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**📋 Dataset Preview**\nExplore the student data — see raw records, stats, and missing value info.")
    with col2:
        st.info("**📊 Visualizations**\nInteractive charts showing how each factor affects student performance.")
    with col3:
        st.info("**🔮 Prediction**\nEnter a student's details and instantly get their predicted score and pass/fail result.")

    col4, col5 = st.columns(2)
    with col4:
        st.success("**📈 Model Performance**\nSee how accurate the ML models are with metrics like R² and accuracy score.")
    with col5:
        st.success("**ℹ️ About**\nLearn about the project, the tech stack, and how everything works.")

    st.markdown("---")
    st.markdown("### 📌 Quick Stats")

    df = load_data()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students",    f"{len(df):,}")
    m2.metric("Pass Rate",         f"{(df['pass_fail']=='Pass').mean():.1%}")
    m3.metric("Avg Study Hours",   f"{df['study_hours'].mean():.1f} hrs")
    m4.metric("Avg Final Score",   f"{df['final_score'].mean():.1f} / 100")

    st.markdown('<div class="tip-box">💡 <b>Tip:</b> Use the sidebar on the left to navigate between pages.</div>',
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 2 — DATASET PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Dataset Preview":
    st.markdown('<p class="section-header">📋 Dataset Preview</p>', unsafe_allow_html=True)

    df = load_data()

    tab1, tab2, tab3 = st.tabs(["📄 Raw Data", "📊 Statistics", "🔍 Missing Values"])

    with tab1:
        n = st.slider("Rows to show", 5, 100, 20)
        st.dataframe(df.head(n), use_container_width=True)
        st.caption(f"Showing {n} of {len(df)} rows · {df.shape[1]} columns")

    with tab2:
        st.markdown("#### Numeric Column Summary")
        st.dataframe(df.describe().round(2), use_container_width=True)
        st.markdown("#### Categorical Columns")
        c1, c2, c3 = st.columns(3)
        c1.write("**internet_access**"); c1.dataframe(df["internet_access"].value_counts())
        c2.write("**extracurricular**"); c2.dataframe(df["extracurricular"].value_counts())
        c3.write("**pass_fail**");       c3.dataframe(df["pass_fail"].value_counts())

    with tab3:
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        missing_df  = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
        st.dataframe(missing_df[missing_df["Missing Count"] > 0], use_container_width=True)
        st.markdown(
            '<div class="tip-box">💡 Missing values in <b>study_hours</b>, <b>sleep_hours</b>, '
            'and <b>attendance_percent</b> are filled with the column <b>median</b> before training.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 3 — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Visualizations":
    st.markdown('<p class="section-header">📊 Data Visualizations</p>', unsafe_allow_html=True)

    df = load_data()
    sns.set_theme(style="whitegrid")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Score Distribution", "Study Hours vs Score", "Feature Correlations", "Pass/Fail Breakdown"]
    )

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df["final_score"].dropna(), bins=25, color="#2c5364", edgecolor="white", alpha=0.85)
        axes[0].set_title("Final Score Distribution", fontweight="bold")
        axes[0].set_xlabel("Final Score")
        axes[0].set_ylabel("Number of Students")

        counts = df["pass_fail"].value_counts()
        colors = ["#28a745", "#dc3545"]
        axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%",
                    colors=colors, startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        axes[1].set_title("Pass / Fail Ratio", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Study hours vs score
        palette = {"Pass": "#28a745", "Fail": "#dc3545"}
        for outcome, grp in df.groupby("pass_fail"):
            axes[0].scatter(grp["study_hours"], grp["final_score"],
                            alpha=0.45, label=outcome, color=palette[outcome], s=25)
        axes[0].set_xlabel("Study Hours"); axes[0].set_ylabel("Final Score")
        axes[0].set_title("Study Hours vs Final Score", fontweight="bold")
        axes[0].legend()

        # Attendance vs score
        for outcome, grp in df.groupby("pass_fail"):
            axes[1].scatter(grp["attendance_percent"], grp["final_score"],
                            alpha=0.45, label=outcome, color=palette[outcome], s=25)
        axes[1].set_xlabel("Attendance %"); axes[1].set_ylabel("Final Score")
        axes[1].set_title("Attendance % vs Final Score", fontweight="bold")
        axes[1].legend()
        plt.tight_layout()
        st.pyplot(fig)

    with tab3:
        # Correlation heatmap (only numeric columns)
        num_df = df[["study_hours","attendance_percent","previous_score",
                     "sleep_hours","final_score"]].dropna()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.5, ax=ax)
        ax.set_title("Correlation Heatmap (Numeric Features)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown(
            '<div class="tip-box">💡 Values closer to <b>1.0</b> mean strong positive correlation '
            '(both go up together). Values near <b>0</b> mean little relationship.</div>',
            unsafe_allow_html=True,
        )

    with tab4:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        palette_pf = {"Pass": "#28a745", "Fail": "#dc3545"}

        # Internet access vs pass/fail
        internet_counts = df.groupby(["internet_access", "pass_fail"]).size().unstack(fill_value=0)
        internet_counts.plot(kind="bar", ax=axes[0], color=["#dc3545","#28a745"],
                             edgecolor="white", width=0.65)
        axes[0].set_title("Internet Access vs Pass/Fail", fontweight="bold")
        axes[0].set_xlabel("Internet Access"); axes[0].tick_params(axis="x", rotation=0)
        axes[0].legend(title="Result")

        # Parental education vs avg score
        edu_score = df.groupby("parental_education")["final_score"].mean().reindex(
            ["None","Primary","Secondary","Bachelor","Master"])
        axes[1].bar(edu_score.index, edu_score.values, color="#2c5364", edgecolor="white", alpha=0.85)
        axes[1].set_title("Parental Education vs Avg Score", fontweight="bold")
        axes[1].set_xlabel("Education Level"); axes[1].set_ylabel("Avg Final Score")
        axes[1].tick_params(axis="x", rotation=20)

        # Extracurricular vs score
        df.boxplot(column="final_score", by="extracurricular", ax=axes[2],
                   patch_artist=True,
                   boxprops=dict(facecolor="#b8daff"),
                   medianprops=dict(color="#004085", linewidth=2))
        axes[2].set_title("Extracurricular vs Final Score", fontweight="bold")
        axes[2].set_xlabel("Extracurricular (No / Yes)")
        plt.suptitle("")
        plt.tight_layout()
        st.pyplot(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 4 — PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Make a Prediction":
    st.markdown('<p class="section-header">🔮 Make a Prediction</p>', unsafe_allow_html=True)
    st.markdown("Fill in the student details below and click **Predict** to see the result.")

    score_model, pass_fail_model = load_models()

    with st.form("prediction_form"):
        st.markdown("#### 📝 Student Details")

        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.slider(
                "📚 Study Hours (per day)", 0.0, 12.0, 5.0, step=0.5,
                help="Average hours the student studies per day")
            attendance = st.slider(
                "🏫 Attendance (%)", 0.0, 100.0, 75.0, step=1.0,
                help="Percentage of classes attended")
            previous_score = st.slider(
                "📄 Previous Exam Score", 0.0, 100.0, 60.0, step=1.0,
                help="Score from the most recent previous exam")
            sleep_hours = st.slider(
                "😴 Sleep Hours (per night)", 3.0, 12.0, 7.0, step=0.5,
                help="Average hours of sleep per night")

        with col2:
            internet = st.selectbox(
                "🌐 Internet Access at Home", ["Yes", "No"],
                help="Does the student have internet access at home?")
            parental_edu = st.selectbox(
                "👨‍👩‍🎓 Parental Education Level",
                ["None", "Primary", "Secondary", "Bachelor", "Master"],
                index=2,
                help="Highest education level of either parent")
            extra = st.selectbox(
                "⚽ Participates in Extracurricular Activities", ["Yes", "No"],
                help="Does the student join sports, clubs, etc.?")

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True)

    # ── Show results ──────────────────────────────────────────────────────────
    if submitted:
        # Input validation
        errors = []
        if study_hours == 0:
            errors.append("Study hours cannot be 0 — at least 0.5 hours required.")
        if errors:
            for e in errors:
                st.error(e)
        else:
            X_input = encode_input(study_hours, attendance, previous_score,
                                   sleep_hours, internet, parental_edu, extra)

            predicted_score    = score_model.predict(X_input)[0]
            predicted_score    = float(np.clip(predicted_score, 0, 100))
            predicted_pass_enc = pass_fail_model.predict(X_input)[0]
            predicted_pass     = "Pass" if predicted_pass_enc == 1 else "Fail"
            pass_proba         = pass_fail_model.predict_proba(X_input)[0][1]  # P(Pass)

            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")

            r1, r2 = st.columns(2)

            with r1:
                st.markdown(
                    f'<div class="result-score">📊 <b>Predicted Final Score</b><br>'
                    f'<span style="font-size:2.2rem;font-weight:700;color:#004085">'
                    f'{predicted_score:.1f} / 100</span></div>',
                    unsafe_allow_html=True,
                )

            with r2:
                if predicted_pass == "Pass":
                    st.markdown(
                        f'<div class="result-pass">✅ <b>Predicted Result: PASS</b><br>'
                        f'<span style="font-size:1rem">Confidence: {pass_proba:.1%}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-fail">❌ <b>Predicted Result: FAIL</b><br>'
                        f'<span style="font-size:1rem">Pass probability: {pass_proba:.1%}</span></div>',
                        unsafe_allow_html=True,
                    )

            # ── Friendly explanation ──────────────────────────────────────────
            st.markdown("### 💬 What does this mean?")

            tips = []
            if study_hours < 4:
                tips.append("📚 **Study more** — students studying 4+ hours/day score significantly higher.")
            if attendance < 70:
                tips.append("🏫 **Improve attendance** — missing classes is strongly linked to lower scores.")
            if sleep_hours < 6:
                tips.append("😴 **Get more sleep** — 7–8 hours of sleep improves concentration and memory.")
            if previous_score < 50:
                tips.append("📄 **Review past material** — a low previous score suggests knowledge gaps to address.")

            if predicted_pass == "Pass":
                st.success(
                    f"🎉 Great news! Based on the inputs, this student is likely to **pass** "
                    f"with an estimated score of **{predicted_score:.1f}**. Keep up the good work!"
                )
            else:
                st.warning(
                    f"⚠️ This student may be at **risk of failing** (predicted score: **{predicted_score:.1f}**). "
                    f"Here are some areas to improve:"
                )

            if tips:
                for tip in tips:
                    st.markdown(f"- {tip}")
            elif predicted_pass == "Pass":
                st.markdown("- The student is performing well across all measured factors. ✅")

            # Score gauge bar
            st.markdown("#### 📉 Score Gauge")
            progress_val = min(int(predicted_score), 100)
            color = "green" if predicted_score >= 50 else "red"
            st.progress(progress_val / 100)
            st.caption(f"Predicted score: {predicted_score:.1f} / 100  {'✅ Pass zone' if predicted_score >= 50 else '❌ Fail zone (below 50)'}")


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 5 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Model Performance":
    st.markdown('<p class="section-header">📈 Model Performance</p>', unsafe_allow_html=True)

    metrics = load_metrics()

    tab1, tab2 = st.tabs(["🔢 Regression (Score)", "🌲 Classification (Pass/Fail)"])

    with tab1:
        st.markdown("### Linear Regression → Final Score Prediction")
        c1, c2 = st.columns(2)
        c1.metric("Mean Absolute Error (MAE)",
                  f"{metrics['linear_regression']['mae']}",
                  help="On average, predictions are off by this many points")
        c2.metric("R² Score",
                  f"{metrics['linear_regression']['r2']}",
                  help="1.0 = perfect. >0.8 = very good for tabular data")

        st.markdown("""
        **Understanding the metrics:**
        - **MAE** (Mean Absolute Error): On average the predicted score is off by ~5 points.
          That's pretty good for a student predictor!
        - **R² Score**: Measures how well the model explains variance in scores.
          0.83 means the model explains 83% of the variation in final scores.
        """)
        st.markdown(
            '<div class="tip-box">💡 R² > 0.80 is generally considered a good fit '
            'for educational/tabular data.</div>',
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("### Random Forest → Pass/Fail Prediction")
        rf = metrics["random_forest"]
        log_acc = metrics["logistic_regression"]["accuracy"]

        c1, c2, c3 = st.columns(3)
        c1.metric("Random Forest Accuracy", f"{rf['accuracy']:.1%}")
        c2.metric("Logistic Regression Accuracy", f"{log_acc:.1%}", help="Comparison model")
        c3.metric("Test Set Size", f"{metrics['test_size']} students")

        st.markdown("#### Classification Report")
        report = rf["report"]
        report_df = pd.DataFrame({
            "Class":     ["Fail", "Pass"],
            "Precision": [round(report["Fail"]["precision"],3), round(report["Pass"]["precision"],3)],
            "Recall":    [round(report["Fail"]["recall"],3),    round(report["Pass"]["recall"],3)],
            "F1-Score":  [round(report["Fail"]["f1-score"],3),  round(report["Pass"]["f1-score"],3)],
        })
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        # Confusion matrix
        st.markdown("#### Confusion Matrix")
        cm = np.array(rf["confusion"])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Predicted Fail","Predicted Pass"],
                    yticklabels=["Actual Fail","Actual Pass"],
                    linewidths=0.5, ax=ax)
        ax.set_title("Confusion Matrix — Random Forest", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("""
        **How to read the confusion matrix:**
        - **Top-left**: Correctly predicted Fail (True Negatives)
        - **Bottom-right**: Correctly predicted Pass (True Positives)
        - **Top-right**: Predicted Pass but actually Fail (False Positives)
        - **Bottom-left**: Predicted Fail but actually Pass (False Negatives)
        """)

        # Model comparison bar chart
        st.markdown("#### Model Comparison")
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        models = ["Random Forest", "Logistic Regression"]
        accs   = [rf["accuracy"], log_acc]
        bars   = ax2.barh(models, [a*100 for a in accs], color=["#2c5364","#203a43"], height=0.4)
        ax2.set_xlabel("Accuracy (%)")
        ax2.set_xlim(0, 100)
        for bar, val in zip(bars, accs):
            ax2.text(val*100 + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", fontweight="bold")
        ax2.set_title("Pass/Fail Classification Accuracy", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig2)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE 6 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About Project":
    st.markdown('<p class="section-header">ℹ️ About This Project</p>', unsafe_allow_html=True)

    st.markdown("""
    ## 🎓 Student Performance Predictor

    This is a beginner-friendly **end-to-end Data Science project** that demonstrates
    the full machine learning workflow: data generation → cleaning → visualisation →
    model training → deployment as a web application.

    ---

    ### 🧠 How it works

    **Step 1 — Data**  
    A synthetic dataset of 600 students is generated with realistic features including
    study habits, attendance, parental education, and more.

    **Step 2 — Preprocessing**  
    Missing values are filled with medians. Categorical columns (Yes/No, education level)
    are encoded into numbers so machine learning models can understand them.

    **Step 3 — Modelling**  
    Two models are trained:
    - **Linear Regression** predicts the numeric final score (R² ≈ 0.83)
    - **Random Forest** classifies Pass or Fail (accuracy ≈ 86%)

    **Step 4 — App**  
    Streamlit turns the trained models into an interactive web app where anyone can
    enter student details and receive instant predictions.

    ---

    ### 🛠️ Tech Stack

    | Tool | Purpose |
    |------|---------|
    | Python 3 | Core programming language |
    | Pandas | Data loading and manipulation |
    | NumPy | Numerical operations |
    | Scikit-learn | Machine learning models |
    | Matplotlib / Seaborn | Data visualisation |
    | Joblib | Saving & loading models |
    | Streamlit | Web application |

    ---

    ### 📁 Project Structure

    ```
    student-performance-predictor/
    ├── data/
    │   └── student_data.csv       ← 600-row dataset
    ├── models/
    │   ├── score_model.pkl        ← Linear Regression
    │   ├── pass_fail_model.pkl    ← Random Forest
    │   └── metrics.json           ← Saved performance metrics
    ├── notebooks/
    │   └── analysis.ipynb         ← Exploratory Data Analysis
    ├── app.py                     ← This Streamlit app
    ├── train_model.py             ← Model training script
    ├── generate_data.py           ← Dataset generation script
    ├── requirements.txt
    └── README.md
    ```

    ---

    ### 🔮 Future Improvements

    - Add more features (tutoring hours, part-time job, commute time)
    - Try XGBoost or Neural Networks
    - Add a student comparison tool
    - Deploy to Streamlit Cloud or Hugging Face Spaces
    - Connect to a real dataset (UCI ML Repository)
    - Add PDF report export

    ---

    ### 📜 License
    MIT License — free to use and modify for learning purposes.
    """)

    st.markdown(
        '<div class="tip-box">⭐ If you found this project helpful, consider adding it to your '
        'GitHub portfolio!</div>',
        unsafe_allow_html=True,
    )
