"""
brain/main_brain.py — Resume Architect Brain
LangGraph pipeline: analyze → write_resume → score_ats → generate_pdf

Enhanced with:
  • user_feedback field in JOBState
  • refine_node — targeted single-pass rewrite using user feedback + prior ATS notes
  • refine_app — separate compiled mini-graph (refine_node → generate_pdf)
    exported for use by /refine-resume endpoint in engine.py
"""

import io
import re
import operator
import numpy as np
from typing import Annotated, Any, List, Optional, TypedDict

from groq import Groq
from huggingface_hub import InferenceClient
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    HRFlowable, ListFlowable, ListItem,
)


# ─────────────────────────────────────────────────────────────
# ForModel — reusable chat request schema
# ─────────────────────────────────────────────────────────────
class ForModel(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    Model: Any
    model_name: str = "llama-3.3-70b-versatile"
    system_prompt: str
    user_prompt: str
    temperature: float = 0.4


# ─────────────────────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────────────────────
class JOBState(TypedDict):
    # ── set by engine.py before invoking ──────────────────
    job_desc:        str
    raw_resume:      str
    revision_count:  int
    groq_key:        Optional[str]
    hf_key:          Optional[str]
    user_feedback:   str            # ✅ NEW — human feedback for refine pass

    # ── produced by the graph ──────────────────────────────
    jd_keywords:      List[str]
    similarity:       float
    missing_keywords: List[str]
    ai_resume:        str
    ats_score:        float
    section_scores:   dict
    ats_feedback:     str
    pdf_bytes:        Optional[bytes]
    steps_taken:      Annotated[List[str], operator.add]


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def chat_model(request: ForModel) -> str:
    """Single reusable Groq chat call."""
    response = request.Model.chat.completions.create(
        model=request.model_name,
        temperature=request.temperature,
        messages=[
            {"role": "system", "content": request.system_prompt},
            {"role": "user",   "content": request.user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


def semantic_similarity(text_a: str, text_b: str, hf_key: str) -> float:
    """HF sentence-transformers cosine similarity."""
    client = InferenceClient(token=hf_key)
    emb_a = client.feature_extraction(text_a, model="sentence-transformers/all-MiniLM-L6-v2")
    emb_b = client.feature_extraction(text_b, model="sentence-transformers/all-MiniLM-L6-v2")
    a, b = np.array(emb_a), np.array(emb_b)
    score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return round(score, 4)


def keyword_diff(raw_resume: str, jd_keywords: List[str]) -> List[str]:
    resume_lower = raw_resume.lower()
    return [kw for kw in jd_keywords if kw.lower() not in resume_lower]


def parse_score(text: str) -> float:
    """Extracts number after SCORE: only."""
    match = re.search(r"SCORE:\s*(\d{1,3}(?:\.\d+)?)", text, re.IGNORECASE)
    if match:
        return min(float(match.group(1)), 100.0)
    return 50.0


def parse_section_scores(text: str) -> dict:
    sections = ["Summary", "Skills", "Experience", "Education", "Certifications"]
    result = {}
    for s in sections:
        match = re.search(rf"{s}:\s*(\d{{1,3}})", text, re.IGNORECASE)
        if match:
            result[s] = int(match.group(1))
    return result


# ─────────────────────────────────────────────────────────────
# PDF rendering helpers
# ─────────────────────────────────────────────────────────────
def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "cv_name", parent=base["Title"],
            fontSize=22, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1a1a2e"),
            alignment=TA_CENTER, spaceAfter=2,
        ),
        "contact": ParagraphStyle(
            "cv_contact", parent=base["Normal"],
            fontSize=9, textColor=colors.HexColor("#555566"),
            alignment=TA_CENTER, spaceAfter=8,
        ),
        "section": ParagraphStyle(
            "cv_section", parent=base["Normal"],
            fontSize=11, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#0f3460"),
            spaceBefore=10, spaceAfter=3,
        ),
        "job_title": ParagraphStyle(
            "cv_job_title", parent=base["Normal"],
            fontSize=10, fontName="Helvetica-Bold",
            textColor=colors.HexColor("#16213e"), spaceAfter=2,
        ),
        "bullet": ParagraphStyle(
            "cv_bullet", parent=base["Normal"],
            fontSize=9.5, textColor=colors.HexColor("#2d2d2d"),
            leftIndent=12, spaceAfter=2, leading=13,
        ),
        "body": ParagraphStyle(
            "cv_body", parent=base["Normal"],
            fontSize=9.5, textColor=colors.HexColor("#2d2d2d"),
            spaceAfter=3, leading=13,
        ),
    }


def markdown_to_story(md_text: str, styles: dict) -> list:
    story = []
    lines = md_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if line.startswith("# "):
            story.append(Paragraph(line[2:].strip(), styles["name"]))
        elif line.startswith("## "):
            story.append(Spacer(1, 4))
            story.append(Paragraph(line[3:].strip().upper(), styles["section"]))
            story.append(HRFlowable(
                width="100%", thickness=1.2,
                color=colors.HexColor("#0f3460"), spaceAfter=4,
            ))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), styles["job_title"]))
        elif line.startswith(("- ", "* ")):
            bullets = [line[2:].strip()]
            while i + 1 < len(lines) and lines[i + 1].startswith(("- ", "* ")):
                i += 1
                bullets.append(lines[i][2:].strip())
            items = [
                ListItem(Paragraph(b, styles["bullet"]), leftIndent=16)
                for b in bullets
            ]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=8))
        elif line.strip() == "":
            story.append(Spacer(1, 3))
        else:
            style = styles["contact"] if ("|" in line or "@" in line) else styles["body"]
            story.append(Paragraph(line.strip(), style))

        i += 1
    return story


# ─────────────────────────────────────────────────────────────
# Node 1 — Analyze
# ─────────────────────────────────────────────────────────────
def analyze_node(state: JOBState) -> dict:
    print("▶ [analyze] Extracting keywords and computing similarity …")
    client = Groq(api_key=state["groq_key"])

    kw_raw = chat_model(ForModel(
        Model=client,
        system_prompt=(
            "You are an ATS keyword extraction specialist. "
            "Return ONLY a comma-separated list of the 25 most important "
            "technical and role-specific keywords from the job description. "
            "No preamble, no numbering, no explanation."
        ),
        user_prompt=state["job_desc"],
        temperature=0.2,
    ))

    keywords = [kw.strip() for kw in kw_raw.split(",") if kw.strip()]
    sim      = semantic_similarity(state["raw_resume"], state["job_desc"], state["hf_key"])
    missing  = keyword_diff(state["raw_resume"], keywords)

    print(f"   similarity={sim:.3f}  keywords={len(keywords)}  missing={len(missing)}")
    return {
        "jd_keywords":      keywords,
        "similarity":       sim,
        "missing_keywords": missing,
        "steps_taken":      [f"analyze: similarity={sim:.3f}, missing={len(missing)} keywords"],
    }


# ─────────────────────────────────────────────────────────────
# Node 2 — Write Resume
# ─────────────────────────────────────────────────────────────
def write_resume_node(state: JOBState) -> dict:
    revision = state["revision_count"]
    print(f"▶ [write_resume] Revision #{revision + 1} …")
    client = Groq(api_key=state["groq_key"])

    missing_str    = ", ".join(state["missing_keywords"][:20]) or "none"
    prior_feedback = state.get("ats_feedback", "")
    feedback_block = (
        f"\n\nPREVIOUS CRITIC FEEDBACK — fix these in this revision:\n{prior_feedback}"
        if prior_feedback else ""
    )

    ai_resume = chat_model(ForModel(
        Model=client,
        system_prompt=(
            "You are an elite Resume Architect and ATS optimization expert. "
            "Rewrite the provided resume so it passes ATS filters and reads "
            "compellingly to a human recruiter.\n\n"
            "Rules:\n"
            "  • Keep ALL factual details (companies, dates, education, names) 100% accurate.\n"
            "  • Naturally incorporate the MISSING KEYWORDS listed below.\n"
            "  • Start every bullet with a strong action verb.\n"
            "  • Quantify achievements wherever possible.\n"
            "  • Use this exact Markdown structure:\n"
            "      # Full Name\n"
            "      email | phone | linkedin | location\n"
            "      ## Summary\n"
            "      ## Skills\n"
            "      ## Experience\n"
            "      ### Job Title — Company (Start – End)\n"
            "      - bullet\n"
            "      ## Education\n"
            "      ## Certifications\n"
            "  • Output ONLY the Markdown resume. No commentary."
        ),
        user_prompt=f"""JOB DESCRIPTION:
{state['job_desc']}

ORIGINAL RESUME:
{state['raw_resume']}

MISSING KEYWORDS TO INCORPORATE:
{missing_str}
{feedback_block}

Rewrite the resume now:""",
        temperature=0.35,
    ))

    return {
        "ai_resume":      ai_resume,
        "revision_count": revision + 1,
        "steps_taken":    [f"write_resume: revision #{revision + 1} complete"],
    }


# ─────────────────────────────────────────────────────────────
# Node 3 — ATS Scorer
# ─────────────────────────────────────────────────────────────
def score_ats_node(state: JOBState) -> dict:
    print("▶ [score_ats] Running ATS critic …")
    client = Groq(api_key=state["groq_key"])

    critic_output = chat_model(ForModel(
        Model=client,
        system_prompt=(
            "You are an ATS evaluator and senior recruiter.\n"
            "Score each resume section AND give an overall score.\n"
            "Then list up to 5 specific improvements.\n\n"
            "Respond EXACTLY in this format:\n"
            "Summary: <0-100>\n"
            "Skills: <0-100>\n"
            "Experience: <0-100>\n"
            "Education: <0-100>\n"
            "SCORE: <overall 0-100>\n"
            "FEEDBACK:\n"
            "- <improvement 1>\n"
            "- <improvement 2>\n"
            "- <improvement 3>"
        ),
        user_prompt=f"""JOB DESCRIPTION:
{state['job_desc']}

TAILORED RESUME:
{state['ai_resume']}""",
        temperature=0.2,
    ))

    score          = parse_score(critic_output)
    section_scores = parse_section_scores(critic_output)
    feedback       = re.sub(
        r"(Summary|Skills|Experience|Education|Certifications):\s*\d+\s*\n?", "",
        critic_output
    )
    feedback = re.sub(r"SCORE:\s*\d+(\.\d+)?\s*\n?", "", feedback).strip()

    print(f"   ATS score={score}  sections={section_scores}")
    return {
        "ats_score":      score,
        "section_scores": section_scores,
        "ats_feedback":   feedback,
        "steps_taken":    [f"score_ats: score={score}"],
    }


# ─────────────────────────────────────────────────────────────
# Node 4 — Generate PDF
# ─────────────────────────────────────────────────────────────
def generate_pdf_node(state: JOBState) -> dict:
    print("▶ [generate_pdf] Rendering styled A4 PDF …")
    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        topMargin=1.5 * cm,  bottomMargin=1.5 * cm,
    )
    styles = _build_styles()
    story  = markdown_to_story(state["ai_resume"], styles)
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    print(f"   PDF size={len(pdf_bytes)} bytes")
    return {
        "pdf_bytes":   pdf_bytes,
        "steps_taken": ["generate_pdf: A4 PDF rendered"],
    }


# ─────────────────────────────────────────────────────────────
# Node 5 — Refine (NEW)
# Targeted single-pass rewrite driven by user_feedback
# ─────────────────────────────────────────────────────────────
def refine_node(state: JOBState) -> dict:
    """
    Single focused rewrite pass.
    Injects both the user's plain-English feedback AND any prior ATS critic notes
    so the model addresses both human preference and automated scoring at once.
    Does NOT re-analyze keywords or re-score — just rewrites and re-renders PDF.
    """
    print("▶ [refine] Applying user feedback …")
    client = Groq(api_key=state["groq_key"])

    user_fb  = state.get("user_feedback", "").strip()
    ats_fb   = state.get("ats_feedback", "").strip()
    combined = ""
    if user_fb:
        combined += f"USER FEEDBACK:\n{user_fb}\n\n"
    if ats_fb:
        combined += f"PRIOR ATS CRITIC NOTES (also incorporate):\n{ats_fb}"

    refined = chat_model(ForModel(
        Model=client,
        system_prompt=(
            "You are an elite Resume Architect. "
            "You will receive a resume in Markdown and a set of improvement instructions. "
            "Apply ALL instructions precisely — do not ignore any point.\n\n"
            "Rules:\n"
            "  • Keep ALL factual details (companies, dates, education, names) 100% accurate.\n"
            "  • Preserve the exact Markdown structure of the original.\n"
            "  • Do not add sections that weren't there before.\n"
            "  • Output ONLY the revised Markdown resume. No commentary, no preamble."
        ),
        user_prompt=f"""CURRENT RESUME (Markdown):
{state['ai_resume']}

JOB DESCRIPTION (for context):
{state['job_desc']}

INSTRUCTIONS — apply all of these:
{combined}

Output the improved resume now:""",
        temperature=0.3,
    ))

    return {
        "ai_resume":   refined,
        "steps_taken": ["refine: user feedback applied"],
    }


# ─────────────────────────────────────────────────────────────
# Conditional Edge — retry or generate PDF
# ─────────────────────────────────────────────────────────────
def should_revise(state: JOBState) -> str:
    if state["ats_score"] < 72 and state["revision_count"] < 2:
        print(f"   ↩ Score {state['ats_score']} < 72 — requesting revision …")
        return "revise"
    print(f"   ✓ Score {state['ats_score']} accepted — generating PDF …")
    return "done"


# ─────────────────────────────────────────────────────────────
# Main pipeline graph
# ─────────────────────────────────────────────────────────────
def _build_graph() -> StateGraph:
    g = StateGraph(JOBState)

    g.add_node("analyze",      analyze_node)
    g.add_node("write_resume", write_resume_node)
    g.add_node("score_ats",    score_ats_node)
    g.add_node("generate_pdf", generate_pdf_node)

    g.add_edge(START,          "analyze")
    g.add_edge("analyze",      "write_resume")
    g.add_edge("write_resume", "score_ats")
    g.add_edge("generate_pdf", END)

    g.add_conditional_edges(
        "score_ats",
        should_revise,
        {"revise": "write_resume", "done": "generate_pdf"},
    )

    return g.compile()


# ─────────────────────────────────────────────────────────────
# Refine mini-graph (for /refine-resume endpoint)
# ─────────────────────────────────────────────────────────────
def _build_refine_graph() -> StateGraph:
    """
    Lightweight 2-node graph:
      refine_node → generate_pdf_node
    Does NOT run the full analyze/score pipeline.
    """
    g = StateGraph(JOBState)

    g.add_node("refine",       refine_node)
    g.add_node("generate_pdf", generate_pdf_node)

    g.add_edge(START,    "refine")
    g.add_edge("refine", "generate_pdf")
    g.add_edge("generate_pdf", END)

    return g.compile()


brain_app  = _build_graph()
refine_app = _build_refine_graph()