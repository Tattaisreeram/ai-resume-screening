import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import io
import time
import numpy as np

# ============================================================
# 🎨 Streamlit App Configuration
# ============================================================
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main {background-color: #f9fafc;}
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stProgress > div > div > div {
            background-color: #4CAF50;
        }
        .highlight {
            background-color: #fff8b3;
            border-radius: 4px;
            padding: 0px 3px;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 🧠 Utility Functions
# ============================================================

@st.cache_data
def extract_text_from_pdf(file):
    """Extract text content from PDF file using PyPDF2."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += " " + content
        return text.strip()
    except Exception as e:
        st.error(f"❌ Error reading {file.name}: {e}")
        return ""


def preprocess_text(text: str) -> str:
    """Basic text preprocessing for better TF-IDF results."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()


def weighted_rank(job_description: str, resumes: list[str]) -> pd.DataFrame:
    """
    Compute similarity between job description and resumes using
    TF-IDF and cosine similarity, with section-based weighting.
    """
    documents = [job_description] + resumes
    tfidf = TfidfVectorizer(stop_words='english').fit_transform(documents)
    vectors = tfidf.toarray()

    job_vec = vectors[0]
    resume_vecs = vectors[1:]

    similarities = cosine_similarity([job_vec], resume_vecs).flatten()

    df = pd.DataFrame({
        "Resume": [f"Resume_{i+1}.pdf" for i in range(len(resumes))],
        "Base Score": similarities
    })

    # Section weights - emphasize skills, experience keywords
    skill_keywords = ['python', 'javascript', 'fastapi', 'react', 'docker', 'sql', 'aws', 'machine learning']
    exp_keywords = ['developed', 'implemented', 'designed', 'built', 'optimized']

    section_boost = []
    for text in resumes:
        skill_count = sum(kw in text.lower() for kw in skill_keywords)
        exp_count = sum(kw in text.lower() for kw in exp_keywords)
        boost = 0.01 * skill_count + 0.02 * exp_count
        section_boost.append(boost)

    df['Weighted Score'] = df['Base Score'] + np.array(section_boost)
    df['Weighted Score'] = df['Weighted Score'].clip(0, 1)
    df = df.sort_values(by='Weighted Score', ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    df = df[['Rank', 'Resume', 'Base Score', 'Weighted Score']]

    return df


def extract_keywords(text: str, top_n=10):
    """Extract top TF-IDF keywords from text."""
    tfidf = TfidfVectorizer(stop_words="english", max_features=top_n)
    tfidf.fit([text])
    return sorted(tfidf.get_feature_names_out())


# ============================================================
# 🌐 App Layout
# ============================================================

st.title("📄 AI Resume Screening & Candidate Ranking System ")

st.markdown("""
Welcome to the **AI-powered Resume Screening System** built with 
TF-IDF, cosine similarity, and smart keyword weighting.  
This tool ranks resumes based on **how closely they match the Job Description** and even identifies key skill overlaps.
""")

st.sidebar.header("⚙️ Configuration")
weight_skills = st.sidebar.slider("Skill Section Weight", 0.0, 1.0, 0.3)
weight_exp = st.sidebar.slider("Experience Section Weight", 0.0, 1.0, 0.2)
st.sidebar.markdown("---")
st.sidebar.info("Adjust weights to fine-tune matching behavior for different roles.")

# ============================================================
# 📝 Inputs
# ============================================================

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🧠 Job Description")
    job_description = st.text_area(
        "Paste or type the job description here...",
        height=220,
        placeholder="Enter job description with skill requirements, responsibilities, and qualifications..."
    )

with col2:
    st.subheader("📂 Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload multiple resume files (PDF format)",
        type=["pdf"],
        accept_multiple_files=True
    )

# ============================================================
# ⚙️ Main Processing
# ============================================================

if uploaded_files and job_description:
    st.divider()
    st.subheader("🔍 Ranking Resumes")

    progress = st.progress(0)
    resumes_text = []

    for idx, file in enumerate(uploaded_files):
        text = extract_text_from_pdf(file)
        if text:
            resumes_text.append(preprocess_text(text))
        progress.progress((idx + 1) / len(uploaded_files))
        time.sleep(0.1)

    if len(resumes_text) == 0:
        st.warning("⚠️ No valid text could be extracted from the uploaded resumes.")
    else:
        results = weighted_rank(preprocess_text(job_description), resumes_text)
        results["Resume"] = [file.name for file in uploaded_files]

        st.success("✅ Ranking completed successfully!")
        st.write("### 🏆 Top Candidates:")
        st.dataframe(results, use_container_width=True)

        # Highlight best match
        top_resume = results.iloc[0]
        st.markdown(f"""
        <div style='background-color:#e8f5e9;padding:10px;border-radius:8px;margin-top:10px'>
        <b>🥇 Best Match:</b> {top_resume['Resume']}  
        <br>Similarity Score: <b>{round(top_resume['Weighted Score']*100,2)}%</b>
        </div>
        """, unsafe_allow_html=True)

        # Keyword extraction for job description
        with st.expander("🔑 View Key JD Keywords"):
            jd_keywords = extract_keywords(job_description)
            st.write(", ".join([f"`{k}`" for k in jd_keywords]))

        # Download results
        csv_buffer = io.StringIO()
        results.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="resume_ranking_results.csv",
            mime="text/csv"
        )

elif not uploaded_files and not job_description:
    st.info("👆 Please upload resumes and enter a job description to start.")
elif uploaded_files and not job_description:
    st.warning("⚠️ Please enter a job description before ranking resumes.")
elif job_description and not uploaded_files:
    st.warning("⚠️ Please upload at least one PDF resume.")

# ============================================================
# 🧩 Footer
# ============================================================
st.divider()
st.markdown("""
<center>
Developed by <b>Sreeram Tatta</b> 🚀 | AICTE TechSaksham Internship Project  
<br><i>TF-IDF × Cosine Similarity × Streamlit Magic</i>
</center>
""", unsafe_allow_html=True)
