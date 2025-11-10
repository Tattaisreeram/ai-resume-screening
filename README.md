# 📄 AI Resume Screening & Candidate Ranking System

An **AI-powered resume screening web application** built with **Streamlit**, leveraging **TF-IDF** and **cosine similarity** to automatically rank candidates based on how closely their resumes match a given **job description**.
🚀 **Live Demo:** [Click here to open the app](https://ai-resume-screening-g83kj4zxee627rfhrbapox.streamlit.app/)

<p align="center">
  <a href="https://ai-resume-screening-g83kj4zxee627rfhrbapox.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live%20App-Streamlit-brightgreen?style=for-the-badge" alt="Streamlit App">
  </a>
</p>
![App Screenshot](https://github.com/user-attachments/assets/60dfbbc2-aedd-400f-98ca-294ac713b2fd)


---

## 🚀 Features

✅ **Smart Resume Ranking**
- Uses NLP-based similarity scoring (TF-IDF + Cosine Similarity)  
- Automatically identifies the most relevant resumes for a job post

✅ **PDF Parsing**
- Extracts and processes text from uploaded PDF resumes using `PyPDF2`

✅ **Interactive Web UI**
- Built with **Streamlit** — simple, clean, and fast  
- Supports multiple file uploads at once  

✅ **Export Results**
- Displays ranked results in an interactive table  
- Download ranking results as a `.csv` file for recruiters  

✅ **Lightweight & Fast**
- No heavy ML models needed  
- Easy to deploy on **Streamlit Community Cloud**

---

## 🧠 How It Works

1. You input a **Job Description (JD)**  
2. Upload multiple **candidate resumes (PDF format)**  
3. The app:
   - Extracts text from each resume  
   - Vectorizes text using **TF-IDF**  
   - Computes **cosine similarity** between each resume and the JD  
   - Ranks resumes based on highest similarity score  

📈 Output: A ranked table showing which candidate best fits the role.

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| **Frontend / UI** | Streamlit |
| **Backend** | Python |
| **Libraries** | `PyPDF2`, `pandas`, `scikit-learn` |
| **Model** | TF-IDF + Cosine Similarity |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |

---

## 🧰 Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Tattaisreeram/ai-resume-screening.git
cd ai-resume-screening
