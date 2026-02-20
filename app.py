"""
AI Resume Analyzer - Production Grade
======================================
A Streamlit-based ML-powered resume evaluation tool that detects skills,
calculates match scores, and predicts career paths from uploaded PDF resumes.

Author: Abhay
Version: 2.0.0
"""

import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import streamlit as st
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
PAGE_CONFIG = {
    "page_title": "AI Resume Analyzer",
    "page_icon": "🚀",
    "layout": "wide",
}

TRAINING_DATA_PATH = Path("training_data.csv")

SKILLS_CATALOG: list[str] = [
    # Data & Analytics
    "python", "sql", "mysql", "machine learning", "deep learning",
    "data science", "power bi", "excel", "tableau", "pandas", "numpy",
    "scikit-learn",
    # Web Development
    "html", "css", "javascript", "react",
    # Backend Frameworks
    "django", "flask",
]

CSS_STYLES = """
<style>
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0);    }
}

@keyframes glow {
    0%,100% { box-shadow: 0 0 5px  #00c6ff; }
    50%      { box-shadow: 0 0 20px #00c6ff; }
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #141e30, #243b55);
    color: white;
}

.hero-title {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeIn 1.5s ease-in-out;
}

.hero-subtitle {
    text-align: center;
    color: #a0b4c8;
    font-size: 16px;
    margin-bottom: 30px;
    animation: fadeIn 2s ease-in-out;
}

.card {
    background: rgba(255, 255, 255, 0.05);
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    animation: fadeIn 1s ease-in-out;
}

button[kind="primary"] {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    transition: transform 0.3s;
}

button[kind="primary"]:hover {
    animation: glow 1.5s infinite;
    transform: scale(1.05);
}

.skill-tag {
    display: inline-block;
    background: rgba(0, 198, 255, 0.15);
    border: 1px solid #00c6ff;
    color: #00c6ff;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 4px;
    font-size: 13px;
}
</style>
"""

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def render_page_config() -> None:
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def render_hero_section() -> None:
    """Render the top hero banner with title and subtitle."""
    st.markdown(
        """
        <h1 class='hero-title'>🚀 AI Resume Analyzer</h1>
        <p class='hero-subtitle'>ML-Powered Resume Evaluation & Smart Career Prediction</p>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data & Model Layer
# ---------------------------------------------------------------------------

def load_training_data(path: Path) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        DataFrame with at least 'resume_text' and 'category' columns.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found at '{path}'. "
            "Please ensure 'training_data.csv' is in the working directory."
        )

    df = pd.read_csv(path)
    required_columns = {"resume_text", "category"}

    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Training CSV is missing columns: {missing}")

    if df.empty:
        raise ValueError("Training data CSV is empty.")

    logger.info("Training data loaded: %d records", len(df))
    return df


@st.cache_resource(show_spinner="Training ML model…")
def train_career_model() -> tuple[LogisticRegression, TfidfVectorizer]:
    """
    Train a TF-IDF + Logistic Regression model for career path prediction.

    Uses Streamlit's cache so the model is trained only once per session.

    Returns:
        Tuple of (trained LogisticRegression model, fitted TfidfVectorizer).

    Raises:
        FileNotFoundError | ValueError: Propagated from load_training_data.
    """
    df = load_training_data(TRAINING_DATA_PATH)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),   # unigrams + bigrams for richer features
        sublinear_tf=True,    # log-scale TF to reduce impact of high-frequency terms
    )
    X = vectorizer.fit_transform(df["resume_text"])
    y = df["category"]

    model = LogisticRegression(
    solver='lbfgs',
    max_iter=1000
)
    model.fit(X, y)
    logger.info("Model trained successfully.")
    return model, vectorizer


# ---------------------------------------------------------------------------
# Resume Processing
# ---------------------------------------------------------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract raw text content from every page of an uploaded PDF.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        Concatenated plain text from all pages.

    Raises:
        ValueError: If the PDF contains no extractable text.
    """
    text_parts: list[str] = []

    with pdfplumber.open(uploaded_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                logger.warning("Page %d yielded no text.", page_num)

    if not text_parts:
        raise ValueError(
            "No readable text found in the uploaded PDF. "
            "The file may be scanned or image-based."
        )

    return "\n".join(text_parts)


def clean_resume_text(raw_text: str) -> str:
    """
    Normalise raw resume text for downstream NLP tasks.

    Steps:
        1. Lowercase all characters.
        2. Remove non-alphanumeric characters (keep spaces).
        3. Collapse consecutive whitespace into a single space.

    Args:
        raw_text: Original text extracted from the PDF.

    Returns:
        Cleaned, lowercased string ready for feature extraction.
    """
    text = raw_text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_matched_skills(cleaned_text: str, skills_catalog: list[str]) -> list[str]:
    """
    Identify skills present in the cleaned resume text.

    Uses whole-word boundary matching to avoid false positives
    (e.g. 'r' matching inside 'react').

    Args:
        cleaned_text: Preprocessed resume string.
        skills_catalog: Master list of skills to look for.

    Returns:
        Sorted list of detected skill names.
    """
    detected: list[str] = []
    for skill in skills_catalog:
        # Escape special regex chars in multi-word skills (e.g. "scikit-learn")
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, cleaned_text):
            detected.append(skill)
    return sorted(detected)


def calculate_match_score(detected_skills: list[str], total_skills: int) -> float:
    """
    Compute a percentage match score based on detected vs total skills.

    Args:
        detected_skills: Skills found in the resume.
        total_skills: Total number of skills in the catalog.

    Returns:
        Score as a float between 0.0 and 100.0, rounded to 2 decimal places.
    """
    if total_skills == 0:
        return 0.0
    return round((len(detected_skills) / total_skills) * 100, 2)


def predict_career_path(
    cleaned_text: str,
    model: LogisticRegression,
    vectorizer: TfidfVectorizer,
) -> tuple[str, float]:
    """
    Predict the most suitable career path for a given resume.

    Args:
        cleaned_text: Preprocessed resume content.
        model: Trained LogisticRegression classifier.
        vectorizer: Fitted TfidfVectorizer.

    Returns:
        Tuple of (predicted_category, confidence_percentage).

    Raises:
        NotFittedError: If the model or vectorizer hasn't been fitted.
    """
    feature_vector = vectorizer.transform([cleaned_text])
    prediction: str = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]
    confidence: float = round(max(probabilities) * 100, 1)
    return prediction, confidence


# ---------------------------------------------------------------------------
# UI Rendering
# ---------------------------------------------------------------------------

def render_skills_card(detected_skills: list[str]) -> None:
    """Render a card displaying all detected skills as styled tags."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### ✅ Detected Skills")

    if detected_skills:
        tags_html = " ".join(
            f"<span class='skill-tag'>{skill}</span>" for skill in detected_skills
        )
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("No skills from the catalog were detected in this resume.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_score_card(score: float) -> None:
    """Render a card with a progress bar showing the resume match score."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Resume Match Score")
    st.progress(int(score))
    st.markdown(f"**{score}% skill match** against our catalog of {len(SKILLS_CATALOG)} skills.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_card(prediction: str, confidence: float) -> None:
    """Render a card showing the ML-predicted career path with confidence."""
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🤖 ML Predicted Career Path")
    st.success(f"**{prediction}**")
    st.caption(f"Model confidence: {confidence}%")
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_results(
    detected_skills: list[str],
    score: float,
    prediction: str,
    confidence: float,
) -> None:
    """
    Lay out the full analysis results in a structured two-column + full-width layout.

    Args:
        detected_skills: List of skills found in the resume.
        score: Match score (0–100).
        prediction: Predicted career category.
        confidence: Model confidence percentage.
    """
    col_left, col_right = st.columns(2)

    with col_left:
        render_skills_card(detected_skills)

    with col_right:
        render_score_card(score)

    render_prediction_card(prediction, confidence)


# ---------------------------------------------------------------------------
# Application Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Main application controller.

    Orchestrates page setup, model loading, file upload handling,
    resume processing, and result rendering.
    """
    render_page_config()
    render_hero_section()

    # --- Model Initialisation ---
    try:
        model, vectorizer = train_career_model()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"⚠️ Model could not be loaded: {exc}")
        st.stop()

    # --- File Upload ---
    uploaded_file = st.file_uploader(
        "📂 Upload Your Resume (PDF only)",
        type=["pdf"],
        help="Supports text-based PDFs. Scanned/image PDFs may not extract correctly.",
    )

    if uploaded_file is None:
        st.info("👆 Upload a PDF resume above to get started.")
        st.stop()

    # --- Resume Processing ---
    with st.spinner("Analysing your resume…"):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
        except ValueError as exc:
            st.error(f"📄 PDF Error: {exc}")
            st.stop()

        cleaned_text = clean_resume_text(raw_text)
        detected_skills = extract_matched_skills(cleaned_text, SKILLS_CATALOG)
        score = calculate_match_score(detected_skills, len(SKILLS_CATALOG))

        try:
            prediction, confidence = predict_career_path(cleaned_text, model, vectorizer)
        except NotFittedError:
            st.error("🤖 The prediction model is not ready. Please restart the app.")
            st.stop()

    # --- Display Results ---
    render_analysis_results(detected_skills, score, prediction, confidence)

    st.markdown("---")
    st.caption("Designed & Developed by Abhay verma 🚀 | AI Resume Analyzer v2.0")


if __name__ == "__main__":
    main()
