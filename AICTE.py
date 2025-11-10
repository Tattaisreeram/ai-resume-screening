import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import time

# ----------------------------
# App Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="centered",
)

# ----------------------------
# Utility Functions
# ----------------------------

@st.cache_data
def extract_text_from_pdf(file):
    """Extracts raw text from a PDF file."""
    try:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
        return text.strip()
    except Exception as e:
        st.error(f"Error reading {file.name}: {e}")
        return ""


def rank_resumes(job_description: str, resumes: list[str]) -> pd.DataFrame:
    """Ranks resumes based on similarity to job description using TF-IDF and cosine similarity."""
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer(stop_words="english").fit_transform(documents)
    vectors = vectorizer.toarray()

    job_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_vector], resume_vectors).flatten()

    df = pd.DataFrame({
        "Resume": [f"Resume_{i+1}.pdf" for i in range(len(resumes))],
        "Score": cosine_similarities
    }).sort_values(by="Score", ascending=False)
    df["Rank"] = range(1, len(df) + 1)
    df = df[["Rank", "Resume", "Score"]]
    return df


# ----------------------------
# Streamlit App UI
# ----------------------------

st.title("📄 AI Resume Screening & Candidate Ranking System")
st.markdown("""
This tool uses **TF-IDF** and **cosine similarity** to automatically rank resumes based on their relevance to a given job description.  
Upload resumes (PDFs) and paste your job description below to get instant ranking results.
""")

st.divider()

# Input: Job Description
st.subheader("🧠 Job Description")
job_description = st.text_area("Paste or type the job description here...", height=200, placeholder="Enter job description...")

# Input: File Upload
st.subheader("📂 Upload Resumes")
uploaded_files = st.file_uploader(
    "Upload multiple resume files (PDF format)",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDFs at once."
)

# ----------------------------
# Processing Logic
# ----------------------------

if uploaded_files and job_description:
    st.divider()
    st.subheader("🔍 Ranking Resumes")

    with st.spinner("Analyzing resumes... Please wait."):
        resumes_text = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            if text:
                resumes_text.append(text)
            time.sleep(0.1)  # Simulate progress

        if len(resumes_text) == 0:
            st.warning("No valid text could be extracted from the uploaded resumes.")
        else:
            results = rank_resumes(job_description, resumes_text)
            results["Resume"] = [file.name for file in uploaded_files]

            st.success("✅ Ranking completed successfully!")
            st.dataframe(results, use_container_width=True)

            # Option to download results
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

# ----------------------------
# Footer
# ----------------------------
st.divider()
st.markdown(
    "<center>Developed by <b>Sreeram Tatta</b> | AI-powered Resume Screening System</center>",
    unsafe_allow_html=True
)
