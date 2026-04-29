"""
frontend/front.py — Resume Architect Engine UI

Enhanced with:
  • Profile photo quality check panel (from X-Photo-* headers)
  • In-app PDF preview via pdf2image — no download required to inspect
  • Feedback refinement loop — user types what to improve → /refine-resume
    returns a new PDF, preview updates, download button refreshes
"""

import ast
import io
import streamlit as st
import requests
import fitz          # PyMuPDF — already installed, zero extra dependencies
from PIL import Image

st.set_page_config(
    page_title="Resume Architect Engine",
    page_icon="🏗️",
    layout="wide",
)
st.title("🏗️ Resume Architect Engine")
st.caption("Upload your CV + paste a Job Description → get an ATS-Optimized resume as a PDF.")


# ── Sidebar: API Keys ─────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Authentication")

    hf_api = st.text_input(
        label="Huggingface API Key",
        placeholder="hf_....",
        type="password",
    )
    groq_api = st.text_input(
        label="Groq API Key",
        placeholder="gsk_....",
        type="password",
        help="Get yours at https://console.groq.com/keys",
    )

    if st.button("Verify Keys", use_container_width=True):
        if not hf_api or not groq_api:
            st.error("Please enter all API Keys first.")
        else:
            with st.spinner("Validating keys..."):
                try:
                    r = requests.post(
                        "http://127.0.0.1:8000/verify-keys",
                        json={"hf_key": hf_api, "groq_key": groq_api},
                    )
                    r.raise_for_status()
                    st.session_state["groq_key"]     = groq_api
                    st.session_state["hf_key"]       = hf_api
                    st.session_state["keys_verified"] = True
                    st.success("✅ All keys verified!")
                except requests.exceptions.RequestException as e:
                    st.error(f"Connection Error: {e}")

    if st.session_state.get("keys_verified"):
        st.success("🔓 Keys active")
    else:
        st.warning("🔒 Keys not verified yet")

    if st.button("🔄 Reset Session", use_container_width=True):
        for key in ["groq_key", "hf_key", "keys_verified",
                    "pdf_bytes", "ai_resume", "job_desc_saved",
                    "ats_score", "similarity", "revisions",
                    "feedback", "steps", "found_kw", "missing_kw",
                    "section_scores"]:
            st.session_state.pop(key, None)
        st.rerun()


# ── Helper — render PDF preview (PyMuPDF — no poppler required) ──
def render_pdf_preview(pdf_bytes: bytes, label: str = ""):
    """
    Renders each PDF page to an image using fitz (PyMuPDF).
    No poppler, no system dependency — fitz is already installed.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if label:
            st.markdown(f"**{label}**")
        for i, page in enumerate(doc):
            mat = fitz.Matrix(2.0, 2.0)   # 2x scale ≈ 144 dpi
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, caption=f"Page {i + 1}", use_column_width=True)
        doc.close()
    except Exception as e:
        st.warning(f"Preview unavailable: {e}. You can still download the PDF below.")


# ── Helper — photo quality panel + upload widget ─────────────
def render_photo_panel(headers):
    """
    Displays the profile photo quality check result.
    If photo is missing or has issues, shows an upload widget so the
    user can provide a better headshot — which is checked and embedded
    into the PDF in-place via /replace-photo.
    """
    photo_found    = headers.get("X-Photo-Found", "false") == "true"
    photo_ok       = headers.get("X-Photo-OK", "false") == "true"
    photo_feedback = headers.get("X-Photo-Feedback", "")
    photo_w        = headers.get("X-Photo-Width", "0")
    photo_h        = headers.get("X-Photo-Height", "0")

    icon  = "✅" if photo_ok else ("⚠️" if photo_found else "❌")
    color = "green" if photo_ok else "orange"

    with st.expander(f"{icon} Profile Photo Check", expanded=not photo_ok):
        if photo_found:
            col_a, col_b = st.columns(2)
            col_a.metric("Width",  f"{photo_w} px")
            col_b.metric("Height", f"{photo_h} px")

        st.markdown(
            f"<span style='color:{color}; font-weight:600'>{photo_feedback}</span>",
            unsafe_allow_html=True,
        )

        # ── Upload widget — only shown when photo needs improvement ──
        if not photo_ok:
            if not photo_found:
                st.info(
                    "Many recruiters outside the US/UK expect a professional headshot. "
                    "Upload one below and we will embed it into your CV automatically."
                )
            else:
                st.info(
                    "Tips for a good headshot: square or portrait crop, "
                    "at least 150×150 px, neutral background, well-lit face."
                )

            st.markdown("**📸 Upload a better photo**")
            new_photo = st.file_uploader(
                "JPG, PNG or WEBP — will be quality-checked before embedding",
                type=["jpg", "jpeg", "png", "webp"],
                key="photo_upload_widget",
            )

            if new_photo is not None:
                # Preview the uploaded image immediately so user can judge it
                preview_col, info_col = st.columns([1, 2])
                with preview_col:
                    st.image(new_photo, caption="Your uploaded photo", width=160)
                with info_col:
                    st.caption(
                        f"File: {new_photo.name}  |  "
                        f"Size: {len(new_photo.getvalue()) // 1024} KB"
                    )

                if st.button("📎 Embed Photo into CV", use_container_width=True, key="embed_photo_btn"):
                    with st.spinner("Checking photo quality and embedding into CV…"):
                        try:
                            resp = requests.post(
                                "http://127.0.0.1:8000/replace-photo",
                                files={
                                    "photo":  (new_photo.name, new_photo.getvalue(), new_photo.type),
                                    "resume": ("resume.pdf", st.session_state["pdf_bytes"], "application/pdf"),
                                },
                                timeout=60,
                            )

                            rh = resp.headers
                            replaced = rh.get("X-Photo-Replaced", "false") == "true"
                            fb       = rh.get("X-Photo-Feedback", "")

                            if resp.status_code == 200 and replaced:
                                # Update PDF in session — preview and download both refresh
                                st.session_state["pdf_bytes"] = resp.content
                                # Update photo headers so expander reflects new state
                                updated_headers = dict(st.session_state.get("photo_headers", {}))
                                updated_headers.update({
                                    "X-Photo-Found":    rh.get("X-Photo-Found", "true"),
                                    "X-Photo-OK":       rh.get("X-Photo-OK", "true"),
                                    "X-Photo-Feedback": fb,
                                    "X-Photo-Width":    rh.get("X-Photo-Width", "0"),
                                    "X-Photo-Height":   rh.get("X-Photo-Height", "0"),
                                })
                                st.session_state["photo_headers"] = updated_headers
                                st.success("✅ Photo embedded! Scroll down to see the updated CV preview.")
                                st.rerun()

                            elif resp.status_code == 200 and not replaced:
                                # Photo failed quality check — show why
                                st.error(f"Photo not embedded — quality check failed: {fb}")
                                st.warning(
                                    "Please upload a clearer photo: "
                                    "portrait crop, ≥150×150 px, good lighting."
                                )
                            else:
                                st.error(f"Server error ({resp.status_code}): {resp.text}")

                        except requests.exceptions.Timeout:
                            st.error("Request timed out. Please try again.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Could not connect to engine: {e}")


# ── Main Interface ────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    job_description = st.text_area(
        label="📋 Paste the Job Description",
        height=280,
        placeholder="Copy-paste the full job posting here...",
    )

with col2:
    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader(
        "PDF only",
        type=["pdf"],
        help="Upload your current resume in PDF format.",
    )


# ── Analyze Button ────────────────────────────────────────────
if st.button("🚀 Architect My Resume", use_container_width=True):

    if not st.session_state.get("keys_verified"):
        st.error("Please verify your API keys in the sidebar first!")

    elif not job_description.strip():
        st.error("Please paste the Job Description first!")

    elif len(job_description.strip()) < 120:
        st.warning("The Job Description seems too short. Paste the full posting for best results.")

    elif uploaded_file is None:
        st.error("Please upload your Resume PDF!")

    else:
        with st.spinner("Analyzing and optimizing your resume… this may take 30–60 seconds."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze-resume",
                    files={
                        "resume": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    data={
                        "job_desc": job_description,
                        "groq_key": st.session_state["groq_key"],
                        "hf_key":   st.session_state["hf_key"],
                    },
                    timeout=180,
                )

                if response.status_code == 200:
                    h = response.headers

                    # Store everything in session_state so the refine loop can access it
                    st.session_state["pdf_bytes"]      = response.content
                    st.session_state["ai_resume"]      = h.get("X-AI-Resume", "")
                    st.session_state["job_desc_saved"] = job_description
                    st.session_state["ats_score"]      = float(h.get("X-ATS-Score", 0))
                    st.session_state["similarity"]     = float(h.get("X-Similarity", 0))
                    st.session_state["revisions"]      = int(h.get("X-Revisions", 0))
                    st.session_state["feedback"]       = h.get("X-Feedback", "")
                    st.session_state["steps"]          = h.get("X-Steps", "").split(" | ")
                    st.session_state["found_kw"]       = [k for k in h.get("X-Found-Keywords", "").split(",") if k]
                    st.session_state["missing_kw"]     = [k for k in h.get("X-Missing-Keywords", "").split(",") if k]
                    st.session_state["photo_headers"]  = dict(h)

                    try:
                        st.session_state["section_scores"] = ast.literal_eval(h.get("X-Section-Scores", "{}"))
                    except Exception:
                        st.session_state["section_scores"] = {}

                    st.success("✅ Resume optimized successfully!")
                else:
                    st.error(f"Backend error ({response.status_code}): {response.text}")

            except requests.exceptions.Timeout:
                st.error("Request timed out. The model may be busy — please try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to engine: {e}")


# ── Results Panel (shown when pdf_bytes is in session) ───────
if st.session_state.get("pdf_bytes"):

    pdf_bytes     = st.session_state["pdf_bytes"]
    ats_score     = st.session_state.get("ats_score", 0)
    similarity    = st.session_state.get("similarity", 0)
    revisions     = st.session_state.get("revisions", 0)
    feedback      = st.session_state.get("feedback", "")
    steps         = st.session_state.get("steps", [])
    found_kw      = st.session_state.get("found_kw", [])
    missing_kw    = st.session_state.get("missing_kw", [])
    section_scores = st.session_state.get("section_scores", {})
    photo_headers = st.session_state.get("photo_headers", {})

    st.divider()

    # ── Photo quality check ───────────────────────────────────
    if photo_headers:
        render_photo_panel(photo_headers)

    # ── Metrics row ───────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 ATS Score",        f"{ats_score:.0f} / 100")
    m2.metric("🔗 CV–JD Similarity", f"{similarity:.1f}%")
    m3.metric("🔄 Revisions Made",   revisions)

    # ── Section-level scores ──────────────────────────────────
    if section_scores:
        with st.expander("📊 Section Score Breakdown", expanded=True):
            for section, score in section_scores.items():
                col_label, col_bar = st.columns([1, 3])
                col_label.write(f"**{section}**")
                col_bar.progress(int(score), text=f"{score}/100")

    # ── Keyword analysis ──────────────────────────────────────
    with st.expander("🔍 Keyword Analysis", expanded=True):
        kw1, kw2 = st.columns(2)
        with kw1:
            st.markdown("**✅ JD Keywords Found**")
            st.write(", ".join(found_kw) if found_kw else "None detected")
        with kw2:
            st.markdown("**⚠️ Keywords Missing**")
            st.write(", ".join(missing_kw) if missing_kw else "None — great alignment!")

    # ── In-app PDF preview ────────────────────────────────────
    st.subheader("👁️ CV Preview")
    render_pdf_preview(pdf_bytes, label="Your optimized resume — scroll to inspect all pages")

    # ── Download button ───────────────────────────────────────
    st.download_button(
        label="⬇️ Download Optimized Resume (PDF)",
        data=pdf_bytes,
        file_name="optimized_resume.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    # ── ATS Critic feedback ───────────────────────────────────
    if feedback:
        with st.expander("💬 ATS Critic Feedback"):
            st.markdown(feedback)

    # ── Pipeline audit trail ──────────────────────────────────
    with st.expander("🪵 Pipeline Steps"):
        for step in steps:
            if step:
                st.text(f"• {step}")

    # ──────────────────────────────────────────────────────────
    # Feedback Refinement Loop
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔁 Not happy? Refine your CV")
    st.caption(
        "Tell us what you'd like changed — tone, sections to remove/add, focus area, "
        "seniority level, anything. The AI will apply your notes in one targeted pass."
    )

    user_feedback = st.text_area(
        label="💬 Your feedback",
        placeholder=(
            "e.g. Make the summary sound more senior. "
            "Remove the internship from 2019. "
            "Add more emphasis on Python and cloud skills."
        ),
        height=120,
        key="user_feedback_input",
    )

    if st.button("🔁 Refine Resume", use_container_width=True):
        if not user_feedback.strip():
            st.warning("Please describe what you'd like improved before refining.")
        elif not st.session_state.get("ai_resume"):
            st.error("CV text not available for refinement. Please re-run the full analysis.")
        else:
            with st.spinner("Applying your feedback… usually takes 20–40 seconds."):
                try:
                    refine_resp = requests.post(
                        "http://127.0.0.1:8000/refine-resume",
                        data={
                            "user_feedback": user_feedback,
                            "ai_resume":     st.session_state["ai_resume"],
                            "job_desc":      st.session_state["job_desc_saved"],
                            "groq_key":      st.session_state["groq_key"],
                            "hf_key":        st.session_state["hf_key"],
                        },
                        timeout=120,
                    )

                    if refine_resp.status_code == 200:
                        # Update session state so preview and download reflect the refined CV
                        st.session_state["pdf_bytes"] = refine_resp.content
                        st.session_state["ai_resume"] = refine_resp.headers.get(
                            "X-AI-Resume", st.session_state["ai_resume"]
                        )
                        st.success("✅ Resume refined! Scroll up to see the updated preview.")
                        st.rerun()   # Re-render page so preview block picks up new pdf_bytes
                    else:
                        st.error(
                            f"Refinement failed ({refine_resp.status_code}): {refine_resp.text}"
                        )

                except requests.exceptions.Timeout:
                    st.error("Refinement timed out. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Could not connect to engine: {e}")