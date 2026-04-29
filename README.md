# 🏗️ Resume Architect Engine

An AI-powered resume optimization pipeline that takes your existing CV and a job description, then produces a fully rewritten, ATS-optimized PDF — scored, critiqued, and refined in a loop until it passes.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **ATS Optimization** | LangGraph pipeline rewrites your CV to match the job description keywords |
| 🔄 **Auto-revision loop** | Scores the output and rewrites again (up to 2×) if score < 72 |
| 📊 **Section-level scoring** | Summary, Skills, Experience, Education each scored 0–100 |
| 🔍 **Keyword gap analysis** | Shows exactly which JD keywords were found vs missing |
| 👁️ **In-app PDF preview** | See your optimized CV without downloading (PyMuPDF, no poppler needed) |
| 📸 **Profile photo check** | Detects photo quality issues (resolution, ratio, brightness) |
| 🖼️ **Photo replacement** | Upload a better headshot — quality-checked then embedded into the PDF |
| 💬 **Feedback refinement** | Type what you want changed → AI applies it in a targeted rewrite pass |

---

## 🗂️ Project Structure

```
Resume-Architect-Engine/
├── backend/
│   └── engine.py          # FastAPI backend — all endpoints
├── brain/
│   └── main_brain.py      # LangGraph pipeline — analyze → write → score → PDF
├── frontend/
│   └── front.py           # Streamlit UI
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/Resume-Architect-Engine.git
cd Resume-Architect-Engine
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the backend
```bash
uvicorn backend.engine:app --reload
```

### 5. Run the frontend (new terminal)
```bash
streamlit run frontend/front.py
```

### 6. Open in browser
Navigate to `http://localhost:8501`

---

## 🔑 API Keys Required

You will be prompted to enter these in the sidebar — they are **never stored on disk**:

| Key | Where to get it |
|---|---|
| **Groq API Key** | https://console.groq.com/keys |
| **Hugging Face API Key** | https://huggingface.co/settings/tokens |

---

## 🧠 Pipeline Overview

```
Upload PDF + Job Description
        │
        ▼
  [analyze_node]
  Extract JD keywords → compute semantic similarity (HF embeddings)
        │
        ▼
  [write_resume_node]
  Rewrite CV with missing keywords injected (Groq LLaMA 3.3 70B)
        │
        ▼
  [score_ats_node]
  Score each section + overall ATS score (Groq)
        │
   score < 72?
   ┌────┴────┐
  yes       no
   │         │
   └─► retry ▼
        [generate_pdf_node]
        Render styled A4 PDF (ReportLab, in memory)
        │
        ▼
   Return PDF + metadata headers to frontend
```

---

## 📦 Requirements

```
fastapi
uvicorn
streamlit
groq
huggingface-hub
langgraph
pymupdf          # fitz — PDF parsing and rendering
Pillow
reportlab
numpy
loguru
python-multipart
```

---

## 📄 License

MIT
