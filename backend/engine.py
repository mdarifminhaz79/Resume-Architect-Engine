"""
backend/engine.py — Resume Architect Engine
Enhanced with:
  • check_profile_photo helper — called inside /analyze-resume, result sent in headers
  • /refine-resume endpoint — targeted rewrite with user feedback
"""

import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fitz
from fastapi import FastAPI, HTTPException, File, Form, UploadFile
from fastapi.responses import Response
from groq import Groq
from huggingface_hub import HfApi
from PIL import Image
from pydantic import BaseModel
from loguru import logger

try:
    from brain.main_brain import brain_app, refine_app
    logger.success("Brain loaded successfully")
except Exception as e:
    logger.error(f"Failed to import brain: {e}")
    brain_app = None
    refine_app = None

app = FastAPI(title="Resume Architect Engine")


# ─────────────────────────────────────────────────────────────
# Key validation
# ─────────────────────────────────────────────────────────────
class Key_Receive(BaseModel):
    hf_key: str
    groq_key: str


def hf_key_validate(key: str):
    try:
        api = HfApi(token=key)
        api.whoami()
        logger.success("HF key validated")
        return True
    except Exception as e:
        logger.error(f"Invalid HF key: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid Huggingface Key: {str(e)}")


def groq_key_validate(key: str):
    try:
        client = Groq(api_key=key)
        client.models.list()
        logger.success("Groq key validated")
        return True
    except Exception as e:
        logger.error(f"Invalid Groq key: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid Groq Key: {str(e)}")


# ─────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "".join([page.get_text() for page in doc])
        logger.success("PDF text extracted")
        return text.strip()
    except Exception as e:
        logger.error(f"PDF parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF Parsing Error: {str(e)}")


# ─────────────────────────────────────────────────────────────
# Profile photo quality check
# ─────────────────────────────────────────────────────────────
def check_profile_photo(pdf_bytes: bytes) -> dict:
    """
    Extracts the first embedded image from the PDF and runs quality checks:
      - Presence
      - Minimum resolution (150×150 px)
      - Aspect ratio (portrait/square: 0.6 ≤ w/h ≤ 1.1)
      - Brightness (mean pixel value: 40–230 range to avoid blown-out/dark photos)
    Returns a plain dict that is safe to include in HTTP headers.
    """
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                for img in page.get_images(full=True):
                    xref = img[0]
                    base = doc.extract_image(xref)
                    image = Image.open(io.BytesIO(base["image"])).convert("RGB")
                    w, h = image.size
                    ratio = w / h

                    # Brightness via mean of grayscale conversion
                    gray        = image.convert("L")
                    brightness  = sum(gray.getdata()) / (w * h)

                    is_large    = w >= 150 and h >= 150
                    is_portrait = 0.6 <= ratio <= 1.1
                    is_bright   = 40 <= brightness <= 230

                    issues = []
                    if not is_large:
                        issues.append(f"too small ({w}×{h}px, need ≥150×150)")
                    if not is_portrait:
                        issues.append("wrong aspect ratio (use portrait/square crop)")
                    if not is_bright:
                        issues.append(
                            "too dark" if brightness < 40 else "overexposed/too bright"
                        )

                    return {
                        "found":      "true",
                        "width":      str(w),
                        "height":     str(h),
                        "brightness": str(round(brightness, 1)),
                        "ok":         "true" if not issues else "false",
                        "feedback":   "Photo looks good ✅" if not issues
                                      else "Photo issues: " + "; ".join(issues),
                    }
    except Exception as e:
        logger.warning(f"Photo check failed: {e}")

    return {
        "found":      "false",
        "width":      "0",
        "height":     "0",
        "brightness": "0",
        "ok":         "false",
        "feedback":   "No profile photo found in CV ⚠️",
    }


# ─────────────────────────────────────────────────────────────
# Header sanitizer — HTTP headers must be single-line ASCII
# ─────────────────────────────────────────────────────────────
def safe_header(value: str) -> str:
    """
    Strip control characters AND encode to latin-1 safe ASCII.
    HTTP headers (h11/starlette) require latin-1; any character outside
    that range (em-dash \u2014, smart quotes, etc.) causes UnicodeEncodeError.
    Strategy: replace common Unicode punctuation with ASCII equivalents,
    then drop anything still outside latin-1.
    """
    # Normalise whitespace
    value = value.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Replace common Unicode typography that LLMs love to produce
    replacements = {
        "\u2014": "-",   # em dash  —
        "\u2013": "-",   # en dash  –
        "\u2019": "'",   # right single quote  '
        "\u2018": "'",   # left single quote  '
        "\u201c": '"',   # left double quote  "
        "\u201d": '"',   # right double quote  "
        "\u2022": "*",   # bullet  •
        "\u2026": "...", # ellipsis  …
        "\u00b7": "*",   # middle dot  ·
        "\u00e9": "e",   # é
        "\u00e0": "a",   # à
    }
    for char, replacement in replacements.items():
        value = value.replace(char, replacement)
    # Final safety net — drop anything still outside latin-1
    value = value.encode("latin-1", errors="ignore").decode("latin-1")
    return value.strip()


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def run():
    return {"status": "Backend running successfully"}


@app.post("/verify-keys")
def receive_api(req: Key_Receive):
    hf_key_validate(req.hf_key)
    groq_key_validate(req.groq_key)
    return {"status": "success", "message": "All keys are valid and authenticated"}


@app.post("/analyze-resume")
async def handle_generate(
    job_desc:  str        = Form(...),
    groq_key:  str        = Form(...),
    hf_key:    str        = Form(...),
    resume:    UploadFile = File(...),
):
    # ── Validate file ─────────────────────────────────────────
    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    pdf_bytes   = await resume.read()
    resume_text = extract_text_from_pdf(pdf_bytes)

    if not resume_text:
        raise HTTPException(status_code=400, detail="The PDF appears to be empty or unreadable.")

    if brain_app is None:
        raise HTTPException(status_code=500, detail="Brain failed to load on startup.")

    # ── Profile photo check ───────────────────────────────────
    photo_info = check_profile_photo(pdf_bytes)
    logger.info(f"Photo check: {photo_info}")

    # ── Invoke brain ──────────────────────────────────────────
    inputs = {
        "job_desc":         job_desc,
        "raw_resume":       resume_text,
        "revision_count":   0,
        "groq_key":         groq_key,
        "hf_key":           hf_key,
        "jd_keywords":      [],
        "similarity":       0.0,
        "missing_keywords": [],
        "ai_resume":        "",
        "ats_score":        0.0,
        "section_scores":   {},
        "ats_feedback":     "",
        "user_feedback":    "",
        "pdf_bytes":        None,
        "steps_taken":      [],
    }

    try:
        result = brain_app.invoke(inputs)
        logger.success(f"Brain completed — ATS score={result.get('ats_score')}")
    except Exception as e:
        logger.error(f"Brain error: {e}")
        raise HTTPException(status_code=500, detail=f"Brain Error: {str(e)}")

    pdf_out = result.get("pdf_bytes")
    if not pdf_out:
        raise HTTPException(status_code=500, detail="PDF generation failed.")

    headers = {
        "X-ATS-Score":        safe_header(str(result.get("ats_score", 0))),
        "X-Similarity":       safe_header(str(round(result.get("similarity", 0) * 100, 1))),
        "X-Revisions":        safe_header(str(result.get("revision_count", 0))),
        "X-Section-Scores":   safe_header(str(result.get("section_scores", {}))),
        "X-Missing-Keywords": safe_header(",".join(result.get("missing_keywords", [])[:15])),
        "X-Found-Keywords":   safe_header(",".join(result.get("jd_keywords", [])[:15])),
        "X-Feedback":         safe_header(result.get("ats_feedback", "")[:500]),
        "X-Steps":            safe_header(" | ".join(result.get("steps_taken", []))),
        "X-AI-Resume":        safe_header(result.get("ai_resume", "")[:2000]),
        # Photo check headers
        "X-Photo-Found":      safe_header(photo_info["found"]),
        "X-Photo-OK":         safe_header(photo_info["ok"]),
        "X-Photo-Feedback":   safe_header(photo_info["feedback"]),
        "X-Photo-Width":      safe_header(photo_info["width"]),
        "X-Photo-Height":     safe_header(photo_info["height"]),
        "Content-Disposition": 'attachment; filename="optimized_resume.pdf"',
        "Access-Control-Expose-Headers": (
            "X-ATS-Score, X-Similarity, X-Revisions, X-Section-Scores, "
            "X-Missing-Keywords, X-Found-Keywords, X-Feedback, X-Steps, "
            "X-AI-Resume, X-Photo-Found, X-Photo-OK, X-Photo-Feedback, "
            "X-Photo-Width, X-Photo-Height"
        ),
    }

    return Response(content=pdf_out, media_type="application/pdf", headers=headers)


# ─────────────────────────────────────────────────────────────
# /refine-resume — targeted rewrite using user feedback
# ─────────────────────────────────────────────────────────────
@app.post("/refine-resume")
async def handle_refine(
    user_feedback: str = Form(...),
    ai_resume:     str = Form(...),
    job_desc:      str = Form(...),
    groq_key:      str = Form(...),
    hf_key:        str = Form(...),
):
    """
    Runs only the refine_node + generate_pdf — not the full pipeline.
    Accepts the current CV text + user free-text feedback and returns
    a new improved PDF.
    """
    if not user_feedback.strip():
        raise HTTPException(status_code=400, detail="Feedback cannot be empty.")

    if refine_app is None:
        raise HTTPException(status_code=500, detail="Refine brain failed to load.")

    inputs = {
        "job_desc":      job_desc,
        "ai_resume":     ai_resume,
        "user_feedback": user_feedback,
        "groq_key":      groq_key,
        "hf_key":        hf_key,
        "ats_feedback":  "",
        "pdf_bytes":     None,
        "steps_taken":   [],
        # Unused by refine path but required by JOBState
        "raw_resume":       "",
        "revision_count":   0,
        "jd_keywords":      [],
        "similarity":       0.0,
        "missing_keywords": [],
        "ats_score":        0.0,
        "section_scores":   {},
    }

    try:
        result = refine_app.invoke(inputs)
        logger.success("Refine brain completed")
    except Exception as e:
        logger.error(f"Refine brain error: {e}")
        raise HTTPException(status_code=500, detail=f"Refine Error: {str(e)}")

    pdf_out = result.get("pdf_bytes")
    if not pdf_out:
        raise HTTPException(status_code=500, detail="Refined PDF generation failed.")

    headers = {
        "X-AI-Resume": safe_header(result.get("ai_resume", "")[:2000]),
        "Content-Disposition": 'attachment; filename="refined_resume.pdf"',
        "Access-Control-Expose-Headers": "X-AI-Resume",
    }

    return Response(content=pdf_out, media_type="application/pdf", headers=headers)


# ─────────────────────────────────────────────────────────────
# /replace-photo — embed a new photo into an existing PDF
# ─────────────────────────────────────────────────────────────
def check_image_quality(img_bytes: bytes, source_label: str = "image") -> dict:
    """
    Runs the same quality checks as check_profile_photo but on raw image bytes
    (JPG/PNG uploaded directly by the user — not extracted from a PDF).
    Returns the same dict shape so the frontend can reuse render_photo_panel logic.
    """
    try:
        image      = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        w, h       = image.size
        ratio      = w / h
        gray       = image.convert("L")
        brightness = sum(gray.getdata()) / (w * h)

        is_large    = w >= 150 and h >= 150
        is_portrait = 0.6 <= ratio <= 1.1
        is_bright   = 40 <= brightness <= 230

        issues = []
        if not is_large:
            issues.append(f"too small ({w}x{h}px, need >=150x150)")
        if not is_portrait:
            issues.append("wrong aspect ratio (use portrait/square crop)")
        if not is_bright:
            issues.append("too dark" if brightness < 40 else "overexposed/too bright")

        return {
            "found":      "true",
            "width":      str(w),
            "height":     str(h),
            "brightness": str(round(brightness, 1)),
            "ok":         "true" if not issues else "false",
            "feedback":   "Photo looks good!" if not issues
                          else "Photo issues: " + "; ".join(issues),
        }
    except Exception as e:
        return {
            "found": "false", "width": "0", "height": "0",
            "brightness": "0", "ok": "false",
            "feedback": f"Could not read image: {e}",
        }


def embed_photo_in_pdf(pdf_bytes: bytes, img_bytes: bytes) -> bytes:
    """
    Inserts the supplied image into the first page of the PDF.

    Strategy:
      1. Find the bounding box of the first image already in the PDF
         (so we drop the new photo in exactly the same spot).
      2. If no existing image is found, place the photo in the top-right
         corner as a sensible default.
      3. Delete the old image xref, insert the new one, save to bytes.

    Uses fitz (PyMuPDF) — no extra dependencies.
    """
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[0]

    # ── Locate where the existing photo sits (if any) ─────────
    existing_rect = None
    old_xref      = None
    for img_info in page.get_images(full=True):
        xref      = img_info[0]
        img_rects = page.get_image_rects(xref)
        if img_rects:
            existing_rect = img_rects[0]
            old_xref      = xref
            break

    # ── Remove old photo from page (not from xref table yet) ──
    if old_xref is not None:
        page.delete_image(old_xref)

    # ── Determine insertion rectangle ─────────────────────────
    if existing_rect:
        rect = existing_rect                # same position & size as before
    else:
        # Default: top-right corner, 3 cm × 3.6 cm (passport-style)
        page_w = page.rect.width
        rect   = fitz.Rect(page_w - 110, 36, page_w - 36, 138)

    # ── Convert uploaded image to PNG bytes for fitz ──────────
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    png_buf = io.BytesIO()
    pil_img.save(png_buf, format="PNG")
    png_bytes = png_buf.getvalue()

    # ── Insert new image ──────────────────────────────────────
    page.insert_image(rect, stream=png_bytes, keep_proportion=True)

    # ── Save to bytes ──────────────────────────────────────────
    out_buf = io.BytesIO()
    doc.save(out_buf, garbage=4, deflate=True)
    doc.close()
    return out_buf.getvalue()


@app.post("/replace-photo")
async def handle_replace_photo(
    photo:  UploadFile = File(...),
    resume: UploadFile = File(...),
):
    """
    Accepts:
      • photo  — the new headshot (JPG / PNG / WEBP)
      • resume — the current optimised PDF (bytes from session_state)

    Returns:
      • New PDF bytes with the photo embedded
      • X-Photo-* headers with quality check results on the NEW photo
      • X-Photo-Replaced: "true" | "false"  — whether the swap was done
    """
    allowed = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    if photo.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{photo.content_type}'. Use JPG, PNG or WEBP.",
        )

    img_bytes = await photo.read()
    pdf_bytes = await resume.read()

    # ── Quality check on the incoming photo ───────────────────
    quality = check_image_quality(img_bytes)
    logger.info(f"Uploaded photo quality: {quality}")

    if quality["ok"] != "true":
        # Return quality feedback without touching the PDF
        headers = {
            "X-Photo-Found":    "true",
            "X-Photo-OK":       "false",
            "X-Photo-Feedback": safe_header(quality["feedback"]),
            "X-Photo-Width":    quality["width"],
            "X-Photo-Height":   quality["height"],
            "X-Photo-Replaced": "false",
            "Access-Control-Expose-Headers": (
                "X-Photo-Found, X-Photo-OK, X-Photo-Feedback, "
                "X-Photo-Width, X-Photo-Height, X-Photo-Replaced"
            ),
        }
        # Return the original PDF unchanged so the frontend can keep showing it
        return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)

    # ── Embed the photo into the PDF ──────────────────────────
    try:
        new_pdf = embed_photo_in_pdf(pdf_bytes, img_bytes)
        logger.success("Photo embedded successfully")
    except Exception as e:
        logger.error(f"Photo embed failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to embed photo: {e}")

    headers = {
        "X-Photo-Found":    "true",
        "X-Photo-OK":       "true",
        "X-Photo-Feedback": safe_header(quality["feedback"]),
        "X-Photo-Width":    quality["width"],
        "X-Photo-Height":   quality["height"],
        "X-Photo-Replaced": "true",
        "Content-Disposition": 'attachment; filename="optimized_resume.pdf"',
        "Access-Control-Expose-Headers": (
            "X-Photo-Found, X-Photo-OK, X-Photo-Feedback, "
            "X-Photo-Width, X-Photo-Height, X-Photo-Replaced"
        ),
    }
    return Response(content=new_pdf, media_type="application/pdf", headers=headers)