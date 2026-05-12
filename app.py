"""
AI Resume Analyzer - Production Grade
======================================
A Streamlit-based ML-powered resume evaluation tool that detects skills,
calculates match scores, and predicts career paths from uploaded PDF resumes.

Author: TEAM DATA MINER
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

import requests
from bs4 import BeautifulSoup

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
    "page_icon": "⛏️",
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

MAX_MISSING_SKILLS = 5

PROFESSIONAL_RESUME_TEMPLATE = """[Full Name]
[Job Title / Target Role]
[City, State | Phone | Email | LinkedIn URL]

PROFESSIONAL SUMMARY
Dedicated [Target Role] with [X] years of experience delivering measurable results in [Industry or Domain]. Proven ability to lead cross-functional teams, improve processes, and drive growth while maintaining strong communication and organizational skills.

EXPERIENCE
Company Name, Location
Role Title | Month Year – Present
- Achieved [Key Result] by [Action Taken].
- Led [Team or Initiative] to improve [Outcome].
- Collaborated with [Stakeholders] to deliver [Project or Result].

Company Name, Location
Role Title | Month Year – Month Year
- Delivered [Impact] through [Task or Project].
- Optimized [Process or System] resulting in [Metric].
- Mentored [Team Members / Interns] and enhanced [Skill or Process].

SKILLS
- Technical Skill 1   - Technical Skill 2
- Technical Skill 3   - Technical Skill 4
- Soft Skill 1        - Soft Skill 2

EDUCATION
Degree, Major
School Name, Graduation Year

CERTIFICATIONS
- Certification 1
- Certification 2

PROJECTS
[Project Title] — Brief description of the project, your role, and results.
"""

CSS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    box-sizing: border-box;
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7);
    }
    50% {
        transform: scale(1.05);
        box-shadow: 0 0 0 10px rgba(59, 130, 246, 0);
    }
}

@keyframes shimmer {
    0% {
        background-position: -200% 0;
    }
    100% {
        background-position: 200% 0;
    }
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #1a1a2e 100%);
    background-attachment: fixed;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    min-height: 100vh;
    line-height: 1.6;
}

[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-section {
    text-align: center;
    padding: 4rem 2rem;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1));
    border-radius: 24px;
    margin-bottom: 3rem;
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: fadeInUp 1s ease-out;
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.05) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

.hero-title {
    font-size: clamp(2.5rem, 5vw, 4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
    position: relative;
    z-index: 1;
    text-shadow: 0 0 40px rgba(96, 165, 250, 0.3);
}

.hero-subtitle {
    font-size: clamp(1rem, 2.5vw, 1.25rem);
    color: #cbd5e1;
    margin-bottom: 2rem;
    position: relative;
    z-index: 1;
    opacity: 0.9;
}

.glass-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    border-color: rgba(255, 255, 255, 0.2);
}

.glass-card:hover::before {
    opacity: 1;
}

.card-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.card-title i {
    color: #60a5fa;
    font-size: 1.25rem;
}

.skill-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-top: 1rem;
}

.skill-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(147, 51, 234, 0.2));
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e2e8f0;
    padding: 0.5rem 1rem;
    border-radius: 50px;
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.skill-tag:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(147, 51, 234, 0.3));
}

.progress-container {
    margin: 1.5rem 0;
}

.progress-bar {
    width: 100%;
    height: 12px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    border-radius: 6px;
    transition: width 1s ease-out;
    position: relative;
}

.progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    animation: shimmer 2s infinite;
}

.score-text {
    text-align: center;
    font-size: 1.25rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-top: 1rem;
}

.prediction-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(59, 130, 246, 0.1));
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.prediction-text {
    font-size: 1.5rem;
    font-weight: 700;
    color: #22c55e;
    margin: 1rem 0;
}

.confidence-text {
    color: #94a3b8;
    font-size: 0.875rem;
}

.stButton button {
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    position: relative;
    overflow: hidden;
}

.stButton button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
}

.stButton button:hover::before {
    left: 100%;
}

.stButton button:active {
    transform: translateY(0);
}

.stTextInput input, .stTextArea textarea {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    color: #e2e8f0;
    padding: 1rem;
    font-size: 1rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #60a5fa;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    background: rgba(255, 255, 255, 0.12);
}

.stFileUploader {
    background: rgba(255, 255, 255, 0.08);
    border: 2px dashed rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.stFileUploader:hover {
    border-color: #60a5fa;
    background: rgba(255, 255, 255, 0.12);
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.metric-card {
    background: rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #60a5fa;
    margin-bottom: 0.5rem;
}

.metric-label {
    color: #94a3b8;
    font-size: 0.875rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stExpander {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    backdrop-filter: blur(10px);
}

.stExpander summary {
    color: #f1f5f9;
    font-weight: 600;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
}

.stExpander summary:hover {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px 16px 0 0;
}

.stSuccess {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-radius: 12px;
    padding: 1rem;
    color: #22c55e;
}

.stInfo {
    background: rgba(59, 130, 246, 0.1);
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 12px;
    padding: 1rem;
    color: #3b82f6;
}

.stError {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 12px;
    padding: 1rem;
    color: #ef4444;
}

.footer {
    text-align: center;
    margin-top: 4rem;
    padding: 2rem;
    color: #64748b;
    font-size: 0.875rem;
}

.footer a {
    color: #60a5fa;
    text-decoration: none;
    transition: color 0.3s ease;
}

.footer a:hover {
    color: #a78bfa;
}

/* Responsive Design */
@media (max-width: 768px) {
    .hero-section {
        padding: 2rem 1rem;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
    }
    
    .glass-card {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .metrics-grid {
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
    }
    
    .stButton button {
        padding: 0.75rem 1.5rem;
        font-size: 0.9rem;
    }
}

@media (max-width: 480px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .glass-card {
        padding: 1rem;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .skill-tags {
        justify-content: center;
    }
}

/* Animation classes for dynamic content */
.fade-in-up {
    animation: fadeInUp 0.6s ease-out;
}

.slide-in-left {
    animation: slideInLeft 0.6s ease-out;
}

.slide-in-right {
    animation: slideInRight 0.6s ease-out;
}

.pulse-animation {
    animation: pulse 2s infinite;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.05);
}

::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.3);
}
</style>
"""

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def render_page_config() -> None:
    """Configure Streamlit page settings and inject custom CSS."""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">', unsafe_allow_html=True)
    st.markdown(CSS_STYLES, unsafe_allow_html=True)


def render_hero_section() -> None:
    """Render the top hero banner with title and subtitle."""
    st.markdown(
        """
        <div class='hero-section'>
            <h1 class='hero-title'><i class='fas fa-rocket'></i> AI Resume Analyzer</h1>
            <p class='hero-subtitle'>ML-Powered Resume Evaluation & Smart Career Prediction</p>
        </div>
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


def get_missing_skills(detected_skills: list[str], skills_catalog: list[str], limit: int = MAX_MISSING_SKILLS) -> list[str]:
    """Return a small set of skills missing from the resume that are in the catalog."""
    return [skill for skill in skills_catalog if skill not in detected_skills][:limit]


def calculate_word_count(raw_text: str) -> int:
    """Count the number of words in the resume text."""
    return len(re.findall(r"\b\w+\b", raw_text))


def calculate_skill_density(cleaned_text: str, detected_skills: list[str]) -> float:
    """Calculate the percentage of resume words that mention catalog skills."""
    total_words = len(cleaned_text.split())
    if total_words == 0:
        return 0.0

    mention_count = sum(
        len(re.findall(r"\b" + re.escape(skill) + r"\b", cleaned_text))
        for skill in detected_skills
    )
    return round((mention_count / total_words) * 100, 2)


def get_career_advice(prediction: str) -> list[str]:
    """Provide role-specific advice for improving the resume."""
    text = prediction.lower()
    if "data" in text:
        return [
            "Highlight quantifiable achievements such as improved accuracy, revenue, or efficiency.",
            "List relevant tools and frameworks clearly under a dedicated skills section.",
            "Use action verbs like analysed, modelled, and automated.",
        ]
    if "web" in text or "frontend" in text:
        return [
            "Emphasize modern web technologies, responsive design, and performance improvements.",
            "Showcase projects with links or short descriptions of user impact.",
            "Mention collaboration with designers, product, or backend teams.",
        ]
    if "backend" in text or "engineer" in text:
        return [
            "Detail backend systems, API design, and scalability improvements.",
            "Include the programming languages and frameworks used for each project.",
            "Describe how you improved reliability, speed, or maintainability.",
        ]
    return [
        "Keep your resume format clean, consistent, and easy to scan.",
        "Use strong accomplishment statements with metrics whenever possible.",
        "Match your skills section to the job description for the role you want.",
    ]


def extract_text_from_url(url: str) -> str:
    """
    Extract text content from a job posting URL.

    Args:
        url: The URL of the job posting.

    Returns:
        Extracted plain text from the webpage.

    Raises:
        ValueError: If unable to fetch or extract text.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        text = soup.get_text()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text
    except Exception as e:
        raise ValueError(f"Unable to extract text from URL: {e}")


def extract_job_skills(job_text: str, skills_catalog: list[str]) -> list[str]:
    """
    Extract skills from job description text.

    Args:
        job_text: Cleaned job description text.
        skills_catalog: List of known skills.

    Returns:
        List of detected skills in the job posting.
    """
    cleaned_job = clean_resume_text(job_text)
    return extract_matched_skills(cleaned_job, skills_catalog)


def compare_resume_to_job(resume_skills: list[str], job_skills: list[str]) -> dict:
    """
    Compare resume skills to job requirements.

    Args:
        resume_skills: Skills detected in resume.
        job_skills: Skills required in job.

    Returns:
        Dict with matching, missing, and extra skills.
    """
    matching = [skill for skill in resume_skills if skill in job_skills]
    missing = [skill for skill in job_skills if skill not in resume_skills]
    extra = [skill for skill in resume_skills if skill not in job_skills]
    return {
        'matching': matching,
        'missing': missing,
        'extra': extra,
        'match_percentage': round((len(matching) / len(job_skills) * 100) if job_skills else 0, 2)
    }


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
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'><i class='fas fa-check-circle'></i> Detected Skills</div>", unsafe_allow_html=True)

    if detected_skills:
        tags_html = "<div class='skill-tags'>" + " ".join(
            f"<span class='skill-tag'><i class='fas fa-code'></i> {skill}</span>" for skill in detected_skills
        ) + "</div>"
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("No skills from the catalog were detected in this resume.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_score_card(score: float) -> None:
    """Render a card with a progress bar showing the resume match score."""
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'><i class='fas fa-chart-line'></i> Resume Match Score</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='progress-container'>
        <div class='progress-bar'>
            <div class='progress-fill' style='width: {score}%'></div>
        </div>
        <div class='score-text'>{score}% skill match</div>
    </div>
    <p style='text-align: center; color: #94a3b8; margin-top: 1rem;'>against our catalog of {len(SKILLS_CATALOG)} skills</p>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_card(prediction: str, confidence: float) -> None:
    """Render a card showing the ML-predicted career path with confidence."""
    st.markdown("<div class='glass-card prediction-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'><i class='fas fa-brain'></i> ML Predicted Career Path</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-text'>{prediction}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence-text'>Model confidence: {confidence}%</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_actionables_card(
    detected_skills: list[str],
    missing_skills: list[str],
    word_count: int,
    skill_density: float,
    advice: list[str],
) -> None:
    """Render resume improvement suggestions and practical next steps."""
    with st.expander("📌 Resume Improvement Suggestions", expanded=True):
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'><i class='fas fa-lightbulb'></i> Quick Insights</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metrics-grid'>
            <div class='metric-card'>
                <div class='metric-value'>{len(detected_skills)}</div>
                <div class='metric-label'>Detected Skills</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{word_count}</div>
                <div class='metric-label'>Word Count</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{skill_density}%</div>
                <div class='metric-label'>Skill Density</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if missing_skills:
            st.markdown("**<i class='fas fa-plus-circle'></i> Skills you may want to add:**", unsafe_allow_html=True)
            st.markdown("\n".join(f"- {skill}" for skill in missing_skills))
        else:
            st.success("Nice work! Your resume already includes all core catalog skills.")

        st.markdown("### <i class='fas fa-arrow-up'></i> Suggested Improvements", unsafe_allow_html=True)
        for tip in advice:
            st.markdown(f"- {tip}")

        st.markdown("</div>", unsafe_allow_html=True)


def render_job_comparison_card(comparison: dict) -> None:
    """Render a card showing resume vs job skills comparison."""
    with st.expander("🔍 Resume vs Job Comparison", expanded=True):
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'><i class='fas fa-search'></i> Job Fit Analysis</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class='metrics-grid'>
            <div class='metric-card'>
                <div class='metric-value'>{len(comparison['matching'])}</div>
                <div class='metric-label'>Matching Skills</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(comparison['missing'])}</div>
                <div class='metric-label'>Missing Skills</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(comparison['extra'])}</div>
                <div class='metric-label'>Extra Skills</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{comparison['match_percentage']}%</div>
                <div class='metric-label'>Job Match %</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if comparison['matching']:
            st.markdown("**<i class='fas fa-check-circle' style='color: #22c55e;'></i> Matching Skills:**", unsafe_allow_html=True)
            st.markdown("<div class='skill-tags'>" + " ".join(f"<span class='skill-tag'><i class='fas fa-check'></i> {skill}</span>" for skill in comparison['matching']) + "</div>", unsafe_allow_html=True)

        if comparison['missing']:
            st.markdown("**<i class='fas fa-times-circle' style='color: #ef4444;'></i> Missing Skills (Consider Adding):**", unsafe_allow_html=True)
            st.markdown("<div class='skill-tags'>" + " ".join(f"<span class='skill-tag' style='background: rgba(239, 68, 68, 0.2); border-color: rgba(239, 68, 68, 0.3);'><i class='fas fa-plus'></i> {skill}</span>" for skill in comparison['missing']) + "</div>", unsafe_allow_html=True)

        if comparison['extra']:
            st.markdown("**<i class='fas fa-info-circle' style='color: #3b82f6;'></i> Extra Skills (Not Required for This Job):**", unsafe_allow_html=True)
            st.markdown("<div class='skill-tags'>" + " ".join(f"<span class='skill-tag' style='background: rgba(156, 163, 175, 0.2); border-color: rgba(156, 163, 175, 0.3);'><i class='fas fa-info'></i> {skill}</span>" for skill in comparison['extra']) + "</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


def render_resume_template_section() -> None:
    """Render a professionally structured resume template with download option."""
    with st.expander("📝 Professional Resume Template", expanded=False):
        st.info(
            "This is a manually designed professional resume structure, not an AI-generated sample. "
            "Use it to organise your experience, skills, and achievements clearly."
        )
        st.text_area("Resume Template", PROFESSIONAL_RESUME_TEMPLATE, height=340)
        st.download_button(
            "<i class='fas fa-download'></i> Download Resume Template",
            PROFESSIONAL_RESUME_TEMPLATE,
            file_name="professional_resume_template.txt",
            mime="text/plain",
        )


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
        "📂 **Upload Your Resume (PDF only)**",
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

    # --- Job Description Input ---
    st.markdown("---")
    job_input = st.text_area(
        "🔗 **Paste Job Description URL or Text**",
        placeholder="Enter job posting URL (e.g., https://...) or paste the job description text here...",
        help="Provide a job link or copy-paste the job description to get tailored resume suggestions."
    )

    job_skills = []
    if job_input:
        with st.spinner("Analysing job description…"):
            try:
                if job_input.startswith('http'):
                    job_text = extract_text_from_url(job_input)
                else:
                    job_text = job_input
                job_skills = extract_job_skills(job_text, SKILLS_CATALOG)
                st.success(f"Extracted {len(job_skills)} skills from job description.")
            except ValueError as exc:
                st.error(f"❌ Job Analysis Error: {exc}")

    # --- Display Results ---
    render_analysis_results(detected_skills, score, prediction, confidence)

    missing_skills = get_missing_skills(detected_skills, SKILLS_CATALOG)
    word_count = calculate_word_count(raw_text)
    skill_density = calculate_skill_density(cleaned_text, detected_skills)
    career_advice = get_career_advice(prediction)

    render_actionables_card(detected_skills, missing_skills, word_count, skill_density, career_advice)

    if job_skills:
        comparison = compare_resume_to_job(detected_skills, job_skills)
        render_job_comparison_card(comparison)

    render_resume_template_section()

    st.markdown("""
    <div class='footer'>
        <p>Designed & Developed by <a href='#' target='_blank'>TEAM DATA MINER</a> <i class='fas fa-rocket'></i> | AI Resume Analyzer v2.0</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
