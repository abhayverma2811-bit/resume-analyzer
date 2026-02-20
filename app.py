import streamlit as st
import pdfplumber
import re

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

st.title("📄 AI Resume Analyzer")
st.markdown("Upload your resume and get skill analysis, score & job recommendation 🚀")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

skills_list = [
    "python", "sql", "mysql", "machine learning", "deep learning",
    "data science", "power bi", "excel", "tableau",
    "html", "css", "javascript", "react",
    "django", "flask", "pandas", "numpy", "scikit-learn"
]

job_roles = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "data science"],
    "Data Analyst": ["sql", "excel", "power bi", "python"],
    "Web Developer": ["html", "css", "javascript", "react"],
    "Backend Developer": ["python", "django", "flask", "sql"]
}

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)
    return found_skills

def calculate_score(detected_skills):
    score = (len(detected_skills) / len(skills_list)) * 100
    return round(score, 2)

def recommend_job(detected_skills):
    best_match = ""
    max_match = 0
    
    for role, skills in job_roles.items():
        match_count = len(set(skills) & set(detected_skills))
        if match_count > max_match:
            max_match = match_count
            best_match = role
    
    return best_match

if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    cleaned_text = clean_text(resume_text)
    detected_skills = extract_skills(cleaned_text)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("✅ Detected Skills")
        if detected_skills:
            for skill in detected_skills:
                st.success(skill)
        else:
            st.warning("No skills detected.")

    with col2:
        score = calculate_score(detected_skills)
        st.subheader("📊 Resume Score")
        st.progress(int(score))
        st.write(f"Score: {score}%")

        missing_skills = list(set(skills_list) - set(detected_skills))
        st.subheader("❌ Missing Skills")
        for skill in missing_skills[:5]:
            st.error(skill)

    st.subheader("🎯 Recommended Job Role")
    recommended_job = recommend_job(detected_skills)
    if recommended_job:
        st.success(recommended_job)
    else:
        st.warning("Not enough data to recommend a job role.")