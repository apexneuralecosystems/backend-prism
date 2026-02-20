from fastapi import FastAPI, HTTPException, Body, Depends, status, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field, AliasChoices
from typing import Optional, Dict, Any, Literal, List, Tuple
import os
import asyncio
import json
import base64
import shutil
from pathlib import Path
from urllib.parse import urlparse, urlencode
import html as html_module
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
import motor.motor_asyncio
import uuid
import tempfile
import io
import PyPDF2
import requests
import pytz
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Import services
from services.email_service import send_demo_request_email, send_interview_form_email, send_interview_feedback_form_email
from services.calendar_service import get_group_free_slots
from services.auth import (
    get_password_hash, verify_password,
    create_access_token, create_refresh_token, verify_token
)
from services.auth_models import (
    UserSignup, UserLogin, OTPVerify, PasswordReset,
    TokenResponse, RefreshTokenRequest
)
from services.otp_service import generate_otp, store_otp, verify_otp
from services.resume_parser import ResumeParser
from services.comparator_agent import ComparatorAgent
from services.payments import create_payment_order, capture_payment_order
from services.razorpay_service import create_razorpay_order, verify_razorpay_signature
from services.linkedin_service import (
    save_social_credential,
    get_active_access_token,
    get_social_credential,
    is_token_expired,
    post_to_linkedin,
    post_to_linkedin_with_document,
)
from services.s3_service import (
    upload_bytes_to_s3,
    upload_file_to_s3,
    download_from_s3_url,
    delete_from_s3_url,
    is_s3_url,
    to_object_url,
    generate_presigned_url,
    get_bucket_name,
)
from services.unified_eye_face_tracker import process_video as face_eye_process_video

# Load environment variables
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
print(f"🔍 Loading .env from: {env_path}")

if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
    print("✅ .env file loaded successfully")
else:
    load_dotenv()

# MongoDB Connection
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("MONGO_URL not found in environment variables")

print(f"🔍 Connecting to MongoDB...")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client["prism_db"]
users_collection = db["users"]
user_data_collection = db["user_data"]
organization_data_collection = db["organization_data"]
organization_teams_collection = db["organization_teams"]
organization_members_collection = db["organization_members"]
open_jobs_collection = db["open-jobs"]
ongoing_jobs_collection = db["ongoing-jobs"]
closed_jobs_collection = db["closed-jobs"]
job_applied_collection = db["job-applied"]
refresh_tokens_collection = db["refresh_tokens"]
interview_webhook_collection = db["interview_webhooks"]
interview_feedback_collection = db["interview-feedback"]
offer_webhook_collection = db["offer_webhooks"]
review_requests_collection = db["review_requests"]
payments_collection = db["payments"]

# Initialize PayPal for apex framework (after environment variables are loaded)
# Note: Since we use MongoDB (not PostgreSQL), apex Client won't work
# We'll use a PostgreSQL connection string just for apex Client initialization
# The actual database operations use MongoDB directly
try:
    from services.payments import set_payment_client
    
    # Initialize PayPal client from environment variables
    paypal_client_id = os.getenv("PAYPAL_CLIENT_ID")
    paypal_client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    paypal_mode = os.getenv("PAYPAL_MODE", "sandbox")
    
    print(f"🔍 [PAYPAL INIT] Checking PayPal credentials...")
    print(f"🔍 [PAYPAL INIT] PAYPAL_CLIENT_ID: {'SET' if paypal_client_id else 'NOT SET'}")
    print(f"🔍 [PAYPAL INIT] PAYPAL_CLIENT_SECRET: {'SET' if paypal_client_secret else 'NOT SET'}")
    print(f"🔍 [PAYPAL INIT] PAYPAL_MODE: {paypal_mode}")
    
    # Ensure PayPal environment variables are set in os.environ (apex reads directly from os.environ)
    if paypal_client_id:
        os.environ["PAYPAL_CLIENT_ID"] = paypal_client_id
    if paypal_client_secret:
        os.environ["PAYPAL_CLIENT_SECRET"] = paypal_client_secret
    if paypal_mode:
        os.environ["PAYPAL_MODE"] = paypal_mode
    
    if paypal_client_id and paypal_client_secret:
        # Try to create apex Client with a dummy PostgreSQL URL (apex needs SQL database)
        # We won't actually use this database - we just need the client for PayPal
        try:
            from apex.payments import set_client
            from apex.client import Client
            
            # Use a dummy PostgreSQL URL (apex Client requires SQL database)
            # This is just for initialization - we use MongoDB for actual data
            # The Client won't actually connect since we don't use it for DB operations
            dummy_postgres_url = "postgresql+asyncpg://dummy:dummy@localhost:5432/dummy"
            print(f"🔍 [PAYPAL INIT] Creating apex Client with dummy PostgreSQL URL...")
            apex_client = Client(
                database_url=dummy_postgres_url,
                secret_key=os.getenv("JWT_ACCESS_SECRET", "default-secret")
            )
            print(f"🔍 [PAYPAL INIT] Apex Client created successfully")
            set_client(apex_client)
            set_payment_client(apex_client)
            print("✅ PayPal client initialized via apex framework")
            print(f"   PayPal Mode: {paypal_mode}")
            print(f"   Client ID: {paypal_client_id[:10]}...")
        except Exception as client_error:
            # If Client creation fails (e.g., no PostgreSQL), create a minimal mock
            print(f"⚠️ [PAYPAL INIT] Could not create apex Client (expected with MongoDB): {client_error}")
            print("   Creating minimal client for PayPal operations...")
            
            class MinimalPayPalClient:
                """Minimal client object for PayPal when apex Client can't be used"""
                pass
            
            minimal_client = MinimalPayPalClient()
            set_payment_client(minimal_client)
            print("✅ PayPal credentials configured (minimal client mode)")
            print(f"   PayPal Mode: {paypal_mode}")
            print(f"   Client ID: {paypal_client_id[:10]}...")
            print("   Note: PayPal will use credentials from environment variables")
    else:
        print("❌ ERROR: PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET not set. Payment features will not work.")
        print("   Please check your .env file at: backend/.env")
        print("   Required variables: PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, PAYPAL_MODE")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize PayPal: {e}")
    import traceback
    traceback.print_exc()

# FastAPI app
app = FastAPI(
    title="PRISM API - Complete Backend",
    version="2.0.0",
    description="Unified backend with Authentication, User Profiles, and Demo Requests"
)

# CORS - Configurable via environment variable
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173,http://localhost:5173,https://prism.apexneural.cloud")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]
print(f"🔍 CORS allowed origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_frontend_base_url() -> str:
    """
    Base URL for shareable links (emails, AI interview, payments, invites, etc.).
    Prefer FRONTEND_PUBLIC_URL; else FRONTEND_URL; else localhost.
    If the result is localhost but BACKEND_URL is a production URL, derive frontend from BACKEND_URL
    (e.g. https://prism.backend.apexneural.cloud -> https://prism.apexneural.cloud) so links work in production
    even when FRONTEND_* vars are not passed into the container.
    """
    u = (os.getenv("FRONTEND_PUBLIC_URL") or os.getenv("FRONTEND_URL") or "http://localhost:5173").rstrip("/")
    backend = (os.getenv("BACKEND_URL") or "").strip()
    # If we resolved to localhost but BACKEND_URL looks like production, derive frontend from BACKEND_URL
    if "localhost" in u and backend and "localhost" not in backend:
        try:
            p = urlparse(backend)
            if p.scheme and p.hostname and ".backend." in p.hostname:
                frontend_host = p.hostname.replace(".backend.", ".", 1)
                u = f"{p.scheme}://{frontend_host}"
        except Exception:
            pass
    return u.rstrip("/")


# Log resolved frontend URL at startup (helps verify FRONTEND_PUBLIC_URL / FRONTEND_URL in Docker)
_frontend_base = get_frontend_base_url()
print(f"🔗 Frontend base URL for shareable links: {_frontend_base} (FRONTEND_PUBLIC_URL={bool(os.getenv('FRONTEND_PUBLIC_URL'))}, FRONTEND_URL={bool(os.getenv('FRONTEND_URL'))}, BACKEND_URL={bool(os.getenv('BACKEND_URL'))})")
if "localhost" in _frontend_base:
    print("⚠️ WARNING: Shareable links will use localhost. Set FRONTEND_PUBLIC_URL or FRONTEND_URL, or BACKEND_URL (e.g. https://prism.backend.apexneural.cloud) to derive it.")

# LinkedIn OAuth (org-only) - state stored in memory for callback
oauth_sessions: Dict[str, Dict[str, Any]] = {}
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID")
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI") or (os.getenv("BACKEND_URL", "").rstrip("/") + "/api/oauth/linkedin/callback")
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"


# Mount Static Files
static_dir = backend_dir / "static"
static_dir.mkdir(exist_ok=True)


def _upload_combined_ai_transcripts_to_s3(
    sections: List[Tuple[str, list]],
    applicant_email: str,
    job_id: str,
) -> Optional[str]:
    """
    Build combined AI transcripts content and upload to S3 only.
    Returns S3 URL or None on failure (e.g. S3 not configured).
    """
    if not sections:
        return None
    try:
        lines = [
            "Previous Rounds – AI Interview Transcripts",
            "=" * 60,
            "",
        ]
        for round_name, transcript in sections:
            lines.append(f"--- {round_name} ---")
            lines.append("")
            for msg in transcript:
                role = msg.get("role", "unknown")
                text = msg.get("text", "")
                ts = msg.get("timestamp", "")
                lines.append(f"[{ts}] {role}: {text}")
            lines.append("")
        content = "\n".join(lines).encode("utf-8")
        url = upload_bytes_to_s3(
            content,
            key_prefix=f"transcripts/{job_id}/{applicant_email.replace('@', '_at_')}",
            file_extension=".txt",
            content_type="text/plain; charset=utf-8",
        )
        return url
    except ValueError as e:
        print(f"⚠️ S3 not configured, transcript not stored: {e}")
        return None
    except Exception as e:
        print(f"⚠️ Error uploading transcript to S3: {e}")
        return None


def _score_label(key: str) -> str:
    """Convert score key to readable label."""
    labels = {
        "technical_configuration": "Technical configuration",
        "technical_customization": "Technical customization",
        "communication_skills": "Communication skills",
        "leadership_abilities": "Leadership abilities",
        "enthusiasm": "Enthusiasm",
        "teamwork": "Teamwork",
        "attitude": "Attitude",
    }
    return labels.get(key, key.replace("_", " ").title())


def _format_one_previous_round(pr: dict, index: int) -> str:
    """Format a single previous round as HTML with all details (scores, notes, outcome, etc.)."""
    round_name = pr.get("round", "Round")
    rtype = pr.get("type", "")
    outcome = pr.get("interview_outcome", "Info not given")
    candidate_attended = pr.get("candidate_attended", "")
    reason = html_module.escape((pr.get("reason") or "")[:2000])
    comments = html_module.escape((pr.get("comments") or "")[:5000])
    interviewer_name = pr.get("interviewer_name", "")
    interviewer_email = pr.get("interviewer_email", "")
    interview_date = pr.get("interview_date", "")
    interview_time = pr.get("interview_time", "")

    parts = [f"<div style=\"margin-bottom: 16px; padding: 12px; background: #f8fafc; border-radius: 8px; border-left: 4px solid #3b82f6;\">"]
    parts.append(f"<p style=\"margin: 0 0 8px 0; font-size: 15px; font-weight: 700; color: #1e40af;\">{index}. {round_name}</p>")

    if rtype == "ai_interview":
        parts.append("<p style=\"margin: 0 0 6px 0; font-size: 13px; color: #64748b;\"><strong>Type:</strong> AI interview</p>")
    if interviewer_name or interviewer_email:
        parts.append(f"<p style=\"margin: 0 0 6px 0; font-size: 13px; color: #475569;\"><strong>Interviewer:</strong> {interviewer_name or 'Info not given'} ({interviewer_email or 'Info not given'})</p>")
    if interview_date or interview_time:
        parts.append(f"<p style=\"margin: 0 0 6px 0; font-size: 13px; color: #475569;\"><strong>Date & time:</strong> {interview_date or ''} {interview_time or ''}</p>")
    if candidate_attended:
        parts.append(f"<p style=\"margin: 0 0 6px 0; font-size: 13px; color: #475569;\"><strong>Candidate attended:</strong> {candidate_attended}</p>")

    parts.append(f"<p style=\"margin: 0 0 8px 0; font-size: 13px; color: #475569;\"><strong>Outcome:</strong> {outcome}</p>")
    if reason:
        parts.append(f"<p style=\"margin: 0 0 8px 0; font-size: 13px; color: #475569;\"><strong>Reason:</strong> {reason}</p>")

    scores = pr.get("scores")
    if scores and isinstance(scores, dict):
        parts.append("<p style=\"margin: 8px 0 4px 0; font-size: 13px; font-weight: 600; color: #334155;\">Scores (1–5):</p>")
        parts.append("<ul style=\"margin: 0 0 8px 0; padding-left: 20px; font-size: 13px; color: #475569;\">")
        for key, val in scores.items():
            if val is not None:
                parts.append(f"<li>{_score_label(key)}: {val}</li>")
        parts.append("</ul>")

    if comments:
        parts.append(f"<p style=\"margin: 8px 0 0 0; font-size: 13px; color: #475569;\"><strong>Notes / comments:</strong> {comments}</p>")

    parts.append("</div>")
    return "".join(parts)


async def _get_previous_rounds_and_transcript(job_id: str, applicant_email: str, base_static_dir: Path) -> tuple:
    """
    Fetch previous rounds and AI transcript for a candidate+job.
    Returns: (previous_rounds_section_html, is_first_round, transcript_s3_url_or_none)
    Transcript is stored in S3 only; no local file.
    """
    try:
        applied = await job_applied_collection.find_one(
            {"job_id": job_id, "email": applicant_email}
        )
        previous_rounds = (applied or {}).get("previous_rounds") or []

        if not previous_rounds:
            return (
                "<p><strong>Previous rounds:</strong> This is the first round.</p>",
                True,
                None,
            )

        lines = ["<p style=\"margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #334155;\">Details of previous rounds:</p>"]
        ai_transcript_sections: List[Tuple[str, list]] = []

        for idx, pr in enumerate(previous_rounds, 1):
            rtype = pr.get("type", "")
            feedback_id = pr.get("feedback_id")
            round_name = pr.get("round", "Round")
            fb = None

            if rtype == "ai_interview" and feedback_id:
                fb = await interview_feedback_collection.find_one({"feedback_id": feedback_id})
                if fb and fb.get("transcript"):
                    ai_transcript_sections.append((round_name, fb["transcript"]))

            lines.append(_format_one_previous_round(pr, idx))

            # Include AI interview LLM evaluation (score + suggestion) for interviewer emails
            if rtype == "ai_interview" and fb and fb.get("llm_evaluation"):
                ev = fb["llm_evaluation"]
                score = ev.get("score", "—")
                suggestion = html_module.escape((ev.get("suggestion") or "")[:2000])
                lines.append(
                    "<div style=\"margin: -8px 0 16px 16px; padding: 10px; background: #f0fdf4; "
                    "border-radius: 8px; border-left: 3px solid #22c55e;\">"
                    "<p style=\"margin: 0 0 6px 0; font-size: 13px; font-weight: 600; color: #166534;\">"
                    "Panel evaluation</p>"
                    f"<p style=\"margin: 0 0 4px 0; font-size: 13px; color: #334155;\"><strong>Score:</strong> {score}/100</p>"
                    f"<p style=\"margin: 0; font-size: 13px; color: #475569;\">{suggestion}</p>"
                    "</div>"
                )

        transcript_url = _upload_combined_ai_transcripts_to_s3(
            ai_transcript_sections, applicant_email, job_id
        ) if ai_transcript_sections else None

        return ("".join(lines), False, transcript_url)
    except Exception as e:
        print(f"⚠️ Error fetching previous rounds/transcript: {e}")
        return (
            "<p><strong>Previous rounds:</strong> This is the first round.</p>",
            True,
            None,
        )


def _read_text_from_static_file(static_dir: Path, relative_or_static_path: str) -> str:
    """
    Read a file from static dir and extract text. Supports PDF, DOCX, TXT.
    Path can be e.g. /static/files/jds/x.pdf, static/files/jds/x.pdf, or files/jds/x.pdf.
    Returns '' on failure.
    """
    if not relative_or_static_path or not isinstance(relative_or_static_path, str):
        return ""
    p = relative_or_static_path.strip()
    if p.startswith("/static/"):
        p = p[8:].lstrip("/")
    elif p.startswith("static/"):
        p = p[7:].lstrip("/")
    else:
        p = p.lstrip("/")
    # Ignore full URLs (e.g. https://...)
    if p.startswith("http://") or p.startswith("https://"):
        return ""
    if not p:
        return ""
    full = static_dir / p
    if not full.exists():
        return ""
    try:
        content = full.read_bytes()
        # Try RAG extractor if available (supports PDF, DOCX, etc.)
        try:
            from services.rag import extract_text_from_file
            out = extract_text_from_file(content, full.name)
            if out and not out.startswith("Error:"):
                return out.strip()
        except ImportError:
            pass
        # Fallback: PDF via PyPDF2 or raw decode
        if full.suffix.lower() == ".pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in pdf_reader.pages).strip()
        return content.decode("utf-8", errors="ignore").strip() or ""
    except Exception as e:
        print(f"⚠️ Could not extract text from {full}: {e}")
        return ""


def _read_text_from_static_file_or_s3(static_dir: Path, path_or_url: str) -> str:
    """
    Read file and extract text from S3 URL only (no local storage).
    Supports PDF, DOCX, TXT. Returns '' if not S3 URL or on failure.
    """
    if not path_or_url or not isinstance(path_or_url, str):
        return ""
    path_or_url = path_or_url.strip()
    if not is_s3_url(path_or_url):
        return ""
    try:
        content = download_from_s3_url(path_or_url)
        if not content:
            return ""
        name = path_or_url.split("/")[-1].split("?")[0] or "file.pdf"
        try:
            from services.rag import extract_text_from_file
            out = extract_text_from_file(content, name)
            if out and not out.startswith("Error:"):
                return out.strip()
        except ImportError:
            pass
        if name.lower().endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            return "\n".join((page.extract_text() or "") for page in pdf_reader.pages).strip()
        return content.decode("utf-8", errors="ignore").strip() or ""
    except Exception as e:
        print(f"⚠️ Could not extract text from S3 URL: {e}")
        return ""


def _local_path_or_download_s3(static_dir: Path, path_or_url: str) -> Optional[str]:
    """
    Return a local temp file path for reading/attachment only when path_or_url is an S3 URL.
    Downloads from S3 to a temp file (caller should delete after use).
    Non-S3 paths are not supported (storage is S3-only); returns None.
    """
    if not path_or_url or not isinstance(path_or_url, str):
        return None
    path_or_url = path_or_url.strip()
    if not is_s3_url(path_or_url):
        return None
    try:
        content = download_from_s3_url(path_or_url)
        if not content:
            return None
        import tempfile
        ext = "." + path_or_url.split("/")[-1].split("?")[0].split(".")[-1] if "." in path_or_url.split("/")[-1] else ".bin"
        if len(ext) > 5:
            ext = ".bin"
        fd, path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(path, "wb") as f:
            f.write(content)
        return path
    except Exception as e:
        print(f"⚠️ S3 download for attachment failed: {e}")
        return None


def _build_employees_data_from_org(static_dir: Path, org_data: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build list of parsed-resume dicts from org employees_details for ComparatorAgent.
    Uses parsed_resume_data when present; otherwise fetches resume from URL (S3 or local) and parses.
    """
    if not org_data:
        return []
    employees_details = org_data.get("employees_details", [])
    if not employees_details:
        return []
    parser = ResumeParser()
    result = []
    for emp in employees_details:
        parsed = emp.get("parsed_resume_data") or emp.get("parsedResumeData")
        if isinstance(parsed, dict) and (parsed.get("Name") is not None or parsed.get("Education") is not None):
            result.append(parsed)
            continue
        resume_url = (emp.get("resume_url") or emp.get("resumeUrl") or "").strip()
        if not resume_url:
            continue
        resume_text = _read_text_from_static_file_or_s3(static_dir, resume_url)
        if not resume_text or len(resume_text.strip()) < 10:
            continue
        try:
            parsed = parser.parse(resume_text)
            result.append(parsed)
        except Exception as e:
            print(f"⚠️ [APPLY] Employee resume parse skip: {e}")
    return result


# Strict hiring panel prompt for AI interview evaluation (JD + transcript -> score + suggestion)
AI_INTERVIEW_EVAL_SYSTEM_PROMPT = """You are a highly strict, no-nonsense senior hiring panel with deep domain expertise.

I will provide:
1) A Job Description (JD)
2) An AI interview transcript of a candidate

Your task:
- Evaluate the candidate ONLY based on evidence from the interview transcript.
- Compare the candidate's answers directly against the JD requirements.
- Be extremely strict and critical. Do NOT be lenient.
- Assume this is a high-stakes role with strong competition.
- Penalize:
  - Vague answers
  - Shallow understanding
  - Buzzwords without explanation
  - Missing fundamentals
  - Poor problem-solving clarity
  - Weak communication or uncertainty
- Reward only:
  - Clear reasoning
  - Strong technical depth
  - Real-world understanding
  - Confident and structured answers
  - Direct alignment with the JD

Scoring rules:
- Score out of 100.
- 90–100: Exceptional, clearly hire-ready.
- 70–89: Good but with notable gaps.
- 50–69: Borderline, lacks depth or consistency.
- Below 50: Not suitable for the role.

Output format (STRICTLY follow this JSON format, no extra text):

{
  "score": 50,
  "suggestion": "Explain clearly why the score is low or high, pointing out specific strengths, weaknesses, missing skills, and JD mismatches."
}

Do not add anything outside this JSON."""


async def _evaluate_ai_interview_with_llm(jd_text: str, transcript: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Call OpenAI to evaluate an AI interview transcript against the JD.
    Returns {"score": int, "suggestion": str} or None on failure.
    """
    if not transcript or not jd_text:
        return None
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️ OPENAI_API_KEY not set, skipping AI interview LLM evaluation")
        return None
    lines = []
    for msg in transcript:
        role = msg.get("role", "")
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        label = "Interviewer" if role == "assistant" else "Candidate"
        lines.append(f"{label}: {text}")
    transcript_text = "\n\n".join(lines)
    if not transcript_text.strip():
        return None
    user_content = f"""Job Description:

{jd_text[:15000]}

Interview Transcript:

{transcript_text[:20000]}"""
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_key)
        response = await client.chat.completions.create(
            model=os.getenv("OPENAI_AI_INTERVIEW_EVAL_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": AI_INTERVIEW_EVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        content = (response.choices or [{}])[0].message.content if response.choices else None
        if not content:
            return None
        content = content.strip()
        if content.startswith("```"):
            for sep in ("```json", "```"):
                if content.startswith(sep):
                    content = content[len(sep):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()
        data = json.loads(content)
        score = data.get("score")
        suggestion = data.get("suggestion")
        if score is None or suggestion is None:
            return None
        return {"score": int(score), "suggestion": str(suggestion)}
    except Exception as e:
        print(f"⚠️ AI interview LLM evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Security
security = HTTPBearer()


# ==================== MODELS ====================

class DemoRequest(BaseModel):
    name: str
    email: EmailStr
    phone: str
    companyName: str
    position: str
    date: str
    time: str
    comments: Optional[str] = ""

class BuyCreditsRequest(BaseModel):
    num_credits: int  # Number of credits to buy (1 credit = $10)
    payment_method: Literal["paypal", "razorpay"] = Field(
        default="paypal",
        validation_alias=AliasChoices("payment_method", "paymentMethod"),
    )

class CaptureOrderRequest(BaseModel):
    order_id: str
    org_email: str
    payment_id: Optional[str] = None  # Razorpay payment_id for verify
    signature: Optional[str] = None   # Razorpay signature for verify


# ==================== HELPER FUNCTIONS ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and get current user with org member info"""
    token = credentials.credentials
    payload = verify_token(token, "access")
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    email = payload.get("sub")
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    user = await users_collection.find_one({"email": email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if this is an org member
    member_info = await organization_members_collection.find_one({
        "member_email": email,
        "status": "accepted"
    })
    
    # Enhance user dict with member info
    if member_info:
        user["org_email"] = member_info["org_email"]
        user["role"] = member_info["role"]
        user["is_org_member"] = True
    else:
        # This is the owner/main account
        user["org_email"] = email  # Owner's email is the org email
        user["role"] = "owner"
        user["is_org_member"] = False
    
    return user


# ==================== PERMISSION HELPER FUNCTIONS ====================

def is_org_owner(current_user: dict) -> bool:
    """Check if user is organization owner"""
    return current_user.get("role") == "owner" and not current_user.get("is_org_member", False)

def is_org_member(current_user: dict) -> bool:
    """Check if user is organization member"""
    return current_user.get("is_org_member", False)

def get_org_email(current_user: dict) -> str:
    """Get organization email (owner's email)"""
    return current_user.get("org_email") or current_user.get("email")

def can_edit_job(job: dict, current_user: dict) -> bool:
    """Check if user can edit a specific job"""
    # Owner can edit all jobs in their org
    if is_org_owner(current_user):
        return job.get("company", {}).get("email") == get_org_email(current_user)
    
    # Member can only edit jobs they created
    if is_org_member(current_user):
        return job.get("created_by") == current_user.get("email")
    
    return False


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/signup", status_code=status.HTTP_200_OK)
async def signup(user_data: UserSignup, background_tasks: BackgroundTasks):
    """
    User/Organization Signup - Send OTP to email
    
    IMPORTANT: This endpoint ONLY sends OTP. Account creation happens ONLY
    after successful OTP verification in /api/auth/verify-otp endpoint.
    """
    try:
        # Check if user already exists with ANY user_type (one email = one account)
        existing_user = await users_collection.find_one({
            "email": user_data.email
        })
        
        if existing_user:
            existing_type = existing_user.get("user_type", "user")
            print(f"❌ [SIGNUP] Duplicate email detected: {user_data.email} already registered as {existing_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email already registered as {existing_type}. Each email can only have one account type."
            )
        
        print(f"✅ [SIGNUP] Email {user_data.email} is available for {user_data.user_type} signup")
        
        # Generate and store OTP (NO account creation here)
        otp = generate_otp()
        await store_otp(db, user_data.email, otp)
        
        # Send OTP email in background (non-blocking)
        from services.email_service import send_otp_email
        background_tasks.add_task(send_otp_email, user_data.email, otp, "signup")
        
        return {
            "success": True,
            "message": "OTP sent to email. Please verify to complete signup."
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/verify-otp", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def verify_otp_and_create_user(data: dict = Body(...)):
    """
    Verify OTP and create user account
    
    SECURITY: Account is ONLY created AFTER successful OTP verification.
    If OTP verification fails, no account is created.
    """
    try:
        # Extract fields from request body
        email = data.get("email")
        password = data.get("password")
        name = data.get("name")
        user_type = data.get("user_type", "user")
        otp = data.get("otp")
        
        # Validate required fields
        if not all([email, password, name, otp]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required fields: email, password, name, and otp are required"
            )
        
        # ===== CRITICAL: Verify OTP FIRST - account creation ONLY happens after successful verification =====
        otp_result = await verify_otp(db, email, otp)
        
        # If OTP verification fails, immediately return error - NO account creation
        if not otp_result.get("success", False):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=otp_result.get("message", "OTP verification failed. Account cannot be created.")
            )
        
        # Additional safety check: Ensure OTP verification was definitely successful
        if otp_result.get("success") is not True:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="OTP verification failed. Account cannot be created."
            )
        
        # ===== Only proceed with account creation if OTP verification was successful =====
        
        # Check if user already exists with ANY user_type (one email = one account)
        existing_user = await users_collection.find_one({
            "email": email
        })
        
        if existing_user:
            existing_type = existing_user.get("user_type", "user")
            print(f"❌ [VERIFY-OTP] Duplicate email detected: {email} already registered as {existing_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Email already registered as {existing_type}. Each email can only have one account type."
            )
        
        print(f"✅ [VERIFY-OTP] Email {email} is available, proceeding with {user_type} account creation")
        
        # Only after successful OTP verification, proceed with account creation
        # Hash password
        hashed_password = get_password_hash(password)
        
        # Create user document
        user_doc = {
            "email": email,
            "password": hashed_password,
            "name": name,
            "user_type": user_type,
            "is_verified": True,
            "credits": 1,  # Default credits for new users
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = await users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)
        
        # If organization, create organization data document
        if user_type == "organization":
            org_data = {
                "name": name,
                "email": email,
                "logo_path": "",
                "category": "",
                "company_size": "",
                "website": "",
                "number": "",
                "founded": "",
                "location": "",
                "about_company": "",
                "linkedin_link": "",
                "instagram_link": "",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await organization_data_collection.insert_one(org_data)
        elif user_type == "user":
            # Create user data document for standard user
            user_profile_data = {
                "name": name,
                "user_email": email, # Matching the schema used in user_profile endpoints
                "gender": "",
                "dob": "",
                "location": "",
                "phone": "",
                "summary": "",
                "education": [],
                "skills": [],
                "languages": [],
                "experience": [],
                "certifications": [],
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await user_data_collection.insert_one(user_profile_data)
        
        # Generate tokens
        token_data = {"sub": email, "user_type": user_type}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        # Store refresh token
        await refresh_tokens_collection.insert_one({
            "user_id": user_id,
            "token": refresh_token,
            "created_at": datetime.utcnow()
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user={
                "id": user_id,
                "email": email,
                "name": name,
                "user_type": user_type
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Verify OTP error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/login", response_model=TokenResponse)
async def login(login_data: UserLogin):
    """
    User/Organization Login
    """
    try:
        # Find user
        user = await users_collection.find_one({
            "email": login_data.email,
            "user_type": login_data.user_type
        })
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        if not verify_password(login_data.password, user["password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Check if this is an org member
        member_info = await organization_members_collection.find_one({
            "member_email": login_data.email,
            "status": "accepted"
        })
        
        # Generate tokens with member info
        if member_info:
            token_data = {
                "sub": login_data.email,
                "user_type": login_data.user_type,
                "org_email": member_info["org_email"],
                "role": member_info["role"],
                "is_org_member": True
            }
        else:
            token_data = {
                "sub": login_data.email,
                "user_type": login_data.user_type
            }
        
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        # Store refresh token
        await refresh_tokens_collection.insert_one({
            "user_id": str(user["_id"]),
            "token": refresh_token,
            "created_at": datetime.utcnow()
        })
        
        # Prepare user response
        user_response = {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "user_type": user["user_type"]
        }
        
        # Add member info if applicable
        if member_info:
            user_response["org_email"] = member_info["org_email"]
            user_response["role"] = member_info["role"]
            user_response["is_org_member"] = True
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user_response
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/refresh-token")
async def refresh_access_token(token_request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    """
    try:
        # Verify refresh token
        payload = verify_token(token_request.refresh_token, "refresh")
        
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Check if refresh token exists in database
        token_doc = await refresh_tokens_collection.find_one({
            "token": token_request.refresh_token
        })
        
        if not token_doc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found"
            )
        
        # Generate new access token with all original payload data
        # Preserve org_email, role, and is_org_member for members
        token_data = {
            "sub": payload.get("sub"),
            "user_type": payload.get("user_type")
        }
        
        # Preserve member info if present in original token
        if payload.get("org_email"):
            token_data["org_email"] = payload.get("org_email")
        if payload.get("role"):
            token_data["role"] = payload.get("role")
        if payload.get("is_org_member"):
            token_data["is_org_member"] = payload.get("is_org_member")
        
        access_token = create_access_token(token_data)
        
        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Refresh token error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/logout")
async def logout(token_request: RefreshTokenRequest):
    """
    Logout - Delete refresh token
    """
    try:
        result = await refresh_tokens_collection.delete_one({
            "token": token_request.refresh_token
        })
        
        return {
            "success": True,
            "message": "Logged out successfully"
        }
    
    except Exception as e:
        print(f"❌ Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/forgot-password")
async def forgot_password(email_data: dict = Body(...)):
    """
    Send OTP for password reset
    """
    try:
        email = email_data.get("email")
        user_type = email_data.get("user_type", "user")
        
        # Check if user exists
        user = await users_collection.find_one({
            "email": email,
            "user_type": user_type
        })
        
        if not user:
            # Don't reveal if user exists
            return {
                "success": True,
                "message": "If the email exists, an OTP has been sent"
            }
        
        # Generate and store OTP
        otp = generate_otp()
        await store_otp(db, email, otp)
        
        # Send OTP email
        from services.email_service import send_otp_email
        await send_otp_email(email, otp, "reset")
        
        return {
            "success": True,
            "message": "OTP sent to email"
        }
    
    except Exception as e:
        print(f"❌ Forgot password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/auth/reset-password")
async def reset_password(reset_data: PasswordReset):
    """
    Reset password with OTP
    """
    try:
        # Verify OTP
        otp_result = await verify_otp(db, reset_data.email, reset_data.otp)
        
        if not otp_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=otp_result["message"]
            )
        
        # Hash new password
        hashed_password = get_password_hash(reset_data.new_password)
        
        # Update password
        result = await users_collection.update_one(
            {"email": reset_data.email},
            {"$set": {
                "password": hashed_password,
                "updated_at": datetime.utcnow()
            }}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "success": True,
            "message": "Password reset successful"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Reset password error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== FILE UPLOAD ENDPOINT ====================

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to S3 only. Returns the S3 URL.
    """
    try:
        file_ext = os.path.splitext(file.filename)[1] or ""
        content = await file.read()
        content_type = file.content_type or None
        try:
            file_url = upload_bytes_to_s3(
                content,
                key_prefix="uploads",
                file_extension=file_ext,
                content_type=content_type,
            )
        except ValueError as e:
            print(f"❌ S3 not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="File storage (S3) is not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME and AWS_REGION in .env."
            )
        return {
            "success": True,
            "url": file_url,
            "filename": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


# ==================== RESUME PARSER ENDPOINT ====================

@app.post("/api/parse-resume")
async def parse_resume(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Parse resume PDF and return structured data (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Read file content
        file_content = await file.read()
        
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file provided"
            )
        
        # Extract text from PDF
        parser = ResumeParser()
        try:
            resume_text = parser.extract_text_from_pdf(file_content)
            if not resume_text or len(resume_text.strip()) < 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not extract text from PDF. Please ensure the PDF contains readable text."
                )
        except ValueError as e:
            print(f"❌ Error extracting text from PDF: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to extract text from PDF: {str(e)}"
            )
        
        # Parse resume
        try:
            parsed_data = parser.parse(resume_text)
        except Exception as parse_error:
            print(f"❌ Error parsing resume text: {parse_error}")
            print(f"Resume text length: {len(resume_text)}")
            # Return a basic structure if parsing fails completely
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse resume content. Please try again or check the resume format. Error: {str(parse_error)}"
            )
        
        return {
            "success": True,
            "parsed_data": parsed_data
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error parsing resume: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error parsing resume: {str(e)}"
        )


# ==================== USER PROFILE ENDPOINTS ====================

@app.post("/api/user-profile")
async def create_user_profile(
    profile_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Save candidate profile data to MongoDB (Protected)
    """
    try:
        print(f"🔍 [POST /api/user-profile] Request from: {current_user.get('email')}")
        
        # Remove _id if it exists to prevent duplicate key errors during upsert
        if "_id" in profile_data:
            profile_data.pop("_id")

        # Add user email and name to profile
        profile_data["user_email"] = current_user["email"]
        profile_data["name"] = current_user["name"]
        profile_data["updated_at"] = datetime.utcnow()
        
        # Update existing document or create if not exists
        result = await user_data_collection.update_one(
            {"user_email": current_user["email"]},
            {"$set": profile_data},
            upsert=True
        )
        
        print(f"✅ [POST /api/user-profile] Update result: {result.modified_count} modified, {result.upserted_id} upserted")
        
        return {
            "success": True,
            "message": "Profile saved successfully"
        }
    except Exception as e:
        print(f"❌ Error saving profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving profile: {str(e)}"
        )


@app.get("/api/user-profile")
async def get_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get user's profile data (Protected)
    """
    try:
        profile = await user_data_collection.find_one({
            "user_email": current_user["email"]
        })
        
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Profile not found"
            )
        
        # Convert ObjectId to string
        profile["_id"] = str(profile["_id"])
        
        return {
            "success": True,
            "profile": profile
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== ORGANIZATION PROFILE ENDPOINTS ====================

@app.post("/api/organization-profile")
async def create_organization_profile(
    profile_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Save/Update organization profile data to MongoDB (Protected)
    """
    try:
        print(f"🔍 [POST /api/organization-profile] Request from: {current_user.get('email')}")
        print(f"🔍 [POST /api/organization-profile] Data: {profile_data}")

        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            print(f"❌ [POST /api/organization-profile] User is not organization: {current_user.get('user_type')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Check if user is owner (members cannot edit profile)
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can edit the profile. Members can view but not modify."
            )
        
        # Get org email (for consistency, use get_org_email)
        org_email = get_org_email(current_user)
        
        # Remove _id if it exists
        if "_id" in profile_data:
            profile_data.pop("_id")

        # Update organization data by email
        profile_data["email"] = org_email
        profile_data["updated_at"] = datetime.utcnow()
        
        # Ensure employees_details is an array if provided
        if "employees_details" in profile_data and not isinstance(profile_data["employees_details"], list):
            profile_data["employees_details"] = []
        
        # Update existing document or create if not exists
        result = await organization_data_collection.update_one(
            {"email": org_email},
            {"$set": profile_data},
            upsert=True
        )
        print(f"✅ [POST /api/organization-profile] Update result: {result.modified_count} modified, {result.upserted_id} upserted")
        
        return {
            "success": True,
            "message": "Organization profile saved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving organization profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving organization profile: {str(e)}"
        )


@app.get("/api/organization-profile")
async def get_organization_profile(current_user: dict = Depends(get_current_user)):
    """
    Get organization's profile data (Protected)
    Members see the organization's profile (not their own)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Get org email (for members, this is their org_email, for owners it's their email)
        org_email = get_org_email(current_user)
        
        profile = await organization_data_collection.find_one({
            "email": org_email
        })
        
        if not profile:
            # Return empty profile structure if not found
            return {
                "success": True,
                "profile": {
                    "name": current_user.get("name", ""),
                    "email": current_user["email"],
                    "logo_path": "",
                    "category": "",
                    "company_size": "",
                    "website": "",
                    "number": "",
                    "founded": "",
                    "location": "",
                    "about_company": "",
                    "linkedin_link": "",
                    "instagram_link": ""
                }
            }
        
        # Convert ObjectId to string
        profile["_id"] = str(profile["_id"])
        
        return {
            "success": True,
            "profile": profile
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching organization profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== LINKEDIN (ORG-ONLY) ENDPOINTS ====================

def _generate_oauth_state() -> str:
    """Generate a random state for OAuth."""
    return str(uuid.uuid4()).replace("-", "")[:32]


@app.get("/api/oauth/linkedin/connect")
async def linkedin_connect(
    current_user: dict = Depends(get_current_user),
):
    """
    Start LinkedIn OAuth (org owner only). Stores profile_id = org_email.
    Returns authorization_url for frontend redirect.
    """
    if current_user.get("user_type") != "organization":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only organizations can access this endpoint")
    if not is_org_owner(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only organization owners can connect LinkedIn")
    if not LINKEDIN_CLIENT_ID or not LINKEDIN_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LinkedIn OAuth not configured. Set LINKEDIN_CLIENT_ID and LINKEDIN_CLIENT_SECRET.",
        )
    profile_id = get_org_email(current_user)
    state = _generate_oauth_state()
    oauth_sessions[state] = {
        "profile_id": profile_id,
        "timestamp": datetime.utcnow().isoformat(),
        "platform": "linkedin",
    }
    auth_params = {
        "response_type": "code",
        "client_id": LINKEDIN_CLIENT_ID,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "scope": "openid profile email w_member_social",
        "state": state,
    }
    auth_url = f"{LINKEDIN_AUTH_URL}?{urlencode(auth_params)}"
    return {"success": True, "authorization_url": auth_url}


@app.get("/api/oauth/linkedin/callback")
async def linkedin_callback(code: str, state: str):
    """
    LinkedIn OAuth callback. Exchanges code for token and saves credential by profile_id (org email).
    No auth required (called by LinkedIn redirect).
    """
    if state not in oauth_sessions:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired state. Please try connecting again.")
    session_data = oauth_sessions.pop(state)
    profile_id = session_data.get("profile_id")
    if not profile_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid session.")

    if not LINKEDIN_CLIENT_ID or not LINKEDIN_CLIENT_SECRET:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="LinkedIn OAuth not configured.")

    token_data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET,
    }
    token_response = requests.post(
        LINKEDIN_TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=token_data,
        timeout=10,
    )
    if token_response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"LinkedIn token exchange failed: {token_response.text}",
        )
    token_json = token_response.json()
    access_token = token_json.get("access_token")
    if not access_token:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No access token in LinkedIn response.")

    userinfo_response = requests.get(
        LINKEDIN_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    user_data = userinfo_response.json() if userinfo_response.status_code == 200 else {}
    expires_in = token_json.get("expires_in", 5184000)

    save_social_credential(profile_id, "linkedin", {
        "access_token": access_token,
        "refresh_token": token_json.get("refresh_token"),
        "expires_in": expires_in,
        "platform_user_id": user_data.get("sub"),
        "platform_username": user_data.get("name", "LinkedIn User"),
        "platform_email": user_data.get("email"),
    })

    frontend_base = get_frontend_base_url()
    redirect_url = f"{frontend_base}/organization-profile?linkedin_connected=1"
    return RedirectResponse(url=redirect_url)


@app.get("/api/oauth/linkedin/status")
async def linkedin_status(current_user: dict = Depends(get_current_user)):
    """
    Return whether the organization's LinkedIn is connected (org only).
    Uses org_email (owner's email) as profile_id.
    """
    if current_user.get("user_type") != "organization":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only organizations can access this endpoint")
    profile_id = get_org_email(current_user)
    cred = get_social_credential(profile_id, "linkedin")
    connected = cred is not None and not is_token_expired(cred)
    return {"connected": connected}


@app.post("/api/organization-jobpost/{job_id}/post-linkedin")
async def post_job_to_linkedin(
    job_id: str,
    current_user: dict = Depends(get_current_user),
):
    """
    Post an existing job to LinkedIn using the org owner's connected account (org only).
    """
    if current_user.get("user_type") != "organization":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only organizations can access this endpoint")

    org_email = get_org_email(current_user)
    access_token = get_active_access_token(org_email, "linkedin")
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Connect LinkedIn first in your organization profile, then try again.",
        )

    # Find job in open or ongoing (same org)
    job = await open_jobs_collection.find_one({"job_id": job_id, "company.email": org_email})
    if not job:
        job = await ongoing_jobs_collection.find_one({"job_id": job_id, "company.email": org_email})
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    company_name = job.get("company", {}).get("name", "")
    role = job.get("role", "")
    location = job.get("location", "")
    job_type = job.get("job_type", "full_time")
    openings = job.get("number_of_openings", 1)
    app_close = job.get("application_close_date", "")
    notes = (job.get("notes") or "")[:500]
    file_path = (job.get("file_path") or "").strip()

    frontend_base = get_frontend_base_url()
    job_url = f"{frontend_base}/jobs/{job_id}" if job_id else frontend_base

    # Commentary text (same with or without document attach)
    text_parts = [
        "We're hiring!",
        "",
        f"Role: {role}",
        f"Company: {company_name}",
        f"Location: {location}",
        f"Type: {job_type.replace('_', ' ').title()}",
        f"Openings: {openings}",
    ]
    if app_close:
        text_parts.append(f"Apply by: {app_close}")
    if notes:
        text_parts.append("")
        text_parts.append(notes)
    text_parts.append("")
    text_parts.append(f"Apply here: {job_url}")
    commentary = "\n".join(text_parts)

    # Prefer posting with JD attached as document (LinkedIn Documents API)
    result = None
    if file_path:
        local_jd_path = _local_path_or_download_s3(static_dir, file_path)  # S3 only: downloads to temp for LinkedIn API
        if local_jd_path:
            doc_title = f"JD_{role.replace(' ', '_')[:30]}.pdf" if role else "Job_Description.pdf"
            try:
                result = post_to_linkedin_with_document(
                    access_token=access_token,
                    local_file_path=local_jd_path,
                    commentary=commentary,
                    document_title=doc_title,
                )
                if not result.get("success"):
                    print(f"[LinkedIn] Document post failed (will fallback to text+link): {result.get('error', '')}")
            finally:
                tmp_prefix = tempfile.gettempdir() or "/tmp"
                if local_jd_path and is_s3_url(file_path) and str(local_jd_path).startswith(tmp_prefix):
                    try:
                        os.unlink(local_jd_path)
                    except Exception:
                        pass

    # Fallback: post with text and JD as link (try ARTICLE link card first, then text-only)
    if not result or not result.get("success"):
        jd_url_for_post = None
        if file_path:
            jd_url_for_post = to_object_url(file_path) if is_s3_url(file_path) else (os.getenv("BACKEND_URL") or "http://localhost:5555").rstrip("/") + (file_path if file_path.startswith("/") else f"/{file_path}")
            commentary_with_jd = commentary.replace("Apply here:", f"📎 Job description (PDF): {jd_url_for_post}\n\nApply here:")
        else:
            commentary_with_jd = commentary
        result = post_to_linkedin(access_token=access_token, text=commentary_with_jd, article_url=jd_url_for_post)
        if not result.get("success") and jd_url_for_post:
            result = post_to_linkedin(access_token=access_token, text=commentary_with_jd)

    if not result.get("success"):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=result.get("error", "Failed to post to LinkedIn"),
        )
    return {"success": True, "message": "Job posted to LinkedIn", "post_id": result.get("post_id")}


# ==================== ORGANIZATION MEMBERS ENDPOINTS ====================

class InviteMemberRequest(BaseModel):
    name: str
    email: EmailStr

class BulkInviteMemberRequest(BaseModel):
    members: list[InviteMemberRequest]

@app.post("/api/organization/invite-member")
async def invite_member(
    request: InviteMemberRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Invite a new member to the organization (Owner only)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Check if user is owner
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can invite members"
            )
        
        org_email = current_user["email"]
        
        print(f"🔍 [INVITE-MEMBER] Owner: {org_email}, Inviting: {request.email}")
        
        # Check if member already exists
        existing_member = await organization_members_collection.find_one({
            "org_email": org_email,
            "member_email": request.email
        })
        
        if existing_member:
            member_status = existing_member.get("status", "unknown")
            print(f"❌ [INVITE-MEMBER] Member already exists with status: {member_status}")
            if member_status == "accepted":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This email is already a member of your organization"
                )
            elif member_status == "pending":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="An invitation has already been sent to this email"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Member exists with status: {member_status}"
                )
        
        # Check if email already has ANY account (user or organization)
        existing_user = await users_collection.find_one({
            "email": request.email
        })
        
        if existing_user:
            existing_type = existing_user.get("user_type", "user")
            print(f"❌ [INVITE-MEMBER] Email already registered as: {existing_type}")
            if existing_user.get("email") == org_email:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot invite yourself"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"This email is already registered as {existing_type}. Cannot invite an email that already has an account."
                )
        
        print(f"✅ [INVITE-MEMBER] Email {request.email} is available for invitation")
        
        # Generate unique invite token
        invite_token = str(uuid.uuid4())
        
        # Get organization name
        org_data = await organization_data_collection.find_one({"email": org_email})
        org_name = org_data.get("name", org_email) if org_data else org_email
        
        # Create member record
        member_doc = {
            "org_email": org_email,
            "member_email": request.email,
            "member_name": request.name,
            "role": "member",
            "status": "pending",
            "invite_token": invite_token,
            "invited_by": org_email,
            "invited_at": datetime.utcnow(),
            "accepted_at": None,
            "password_hash": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        await organization_members_collection.insert_one(member_doc)
        
        # Send invitation email in background (non-blocking)
        from services.email_service import send_member_invitation_email
        frontend_url = get_frontend_base_url()
        invite_link = f"{frontend_url}/invite-accept/{invite_token}"
        
        background_tasks.add_task(
            send_member_invitation_email,
            request.email,
            request.name,
            org_name,
            invite_link
        )
        
        print(f"✅ [POST /api/organization/invite-member] Invitation created for {request.email}")
        
        return {
            "success": True,
            "message": "Invitation sent successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error inviting member: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inviting member: {str(e)}"
        )


@app.post("/api/organization/invite-members-bulk")
async def invite_members_bulk(
    request: BulkInviteMemberRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Invite multiple members to the organization at once (Owner only)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Check if user is owner
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can invite members"
            )
        
        org_email = current_user["email"]
        
        # Get organization name
        org_data = await organization_data_collection.find_one({"email": org_email})
        org_name = org_data.get("name", org_email) if org_data else org_email
        frontend_url = get_frontend_base_url()
        
        results = {
            "success": [],
            "failed": []
        }
        
        for member_data in request.members:
            try:
                # Check if member already exists
                existing_member = await organization_members_collection.find_one({
                    "org_email": org_email,
                    "member_email": member_data.email
                })
                
                if existing_member:
                    if existing_member.get("status") == "accepted":
                        results["failed"].append({
                            "email": member_data.email,
                            "reason": "Already a member"
                        })
                        continue
                    elif existing_member.get("status") == "pending":
                        results["failed"].append({
                            "email": member_data.email,
                            "reason": "Invitation already sent"
                        })
                        continue
                
                # Check if email already has ANY account
                existing_user = await users_collection.find_one({
                    "email": member_data.email
                })
                
                if existing_user:
                    if existing_user.get("email") == org_email:
                        results["failed"].append({
                            "email": member_data.email,
                            "reason": "Cannot invite yourself"
                        })
                        continue
                    else:
                        existing_type = existing_user.get("user_type", "user")
                        results["failed"].append({
                            "email": member_data.email,
                            "reason": f"Already registered as {existing_type}"
                        })
                        continue
                
                # Generate unique invite token
                invite_token = str(uuid.uuid4())
                
                # Create member record
                member_doc = {
                    "org_email": org_email,
                    "member_email": member_data.email,
                    "member_name": member_data.name,
                    "role": "member",
                    "status": "pending",
                    "invite_token": invite_token,
                    "invited_by": org_email,
                    "invited_at": datetime.utcnow(),
                    "accepted_at": None,
                    "password_hash": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                
                await organization_members_collection.insert_one(member_doc)
                
                # Send invitation email in background
                invite_link = f"{frontend_url}/invite-accept/{invite_token}"
                from services.email_service import send_member_invitation_email
                background_tasks.add_task(
                    send_member_invitation_email,
                    member_data.email,
                    member_data.name,
                    org_name,
                    invite_link
                )
                
                results["success"].append({
                    "email": member_data.email,
                    "name": member_data.name
                })
                
            except Exception as e:
                print(f"❌ Error inviting {member_data.email}: {e}")
                results["failed"].append({
                    "email": member_data.email,
                    "reason": str(e)
                })
        
        print(f"✅ [POST /api/organization/invite-members-bulk] Invited {len(results['success'])} members, {len(results['failed'])} failed")
        
        return {
            "success": True,
            "message": f"Invited {len(results['success'])} member(s), {len(results['failed'])} failed",
            "results": results
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error in bulk invite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error inviting members: {str(e)}"
        )


class RespondInviteRequest(BaseModel):
    token: str
    action: str  # "accept" or "reject"
    password: Optional[str] = None

@app.post("/api/organization/respond-invite")
async def respond_invite(request: RespondInviteRequest):
    """
    Accept or reject organization invitation (Public endpoint)
    """
    try:
        if request.action not in ["accept", "reject"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Action must be 'accept' or 'reject'"
            )
        
        # Find member record by token
        member_info = await organization_members_collection.find_one({
            "invite_token": request.token,
            "status": "pending"
        })
        
        if not member_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid or expired invitation token"
            )
        
        if request.action == "reject":
            # Update status to rejected
            await organization_members_collection.update_one(
                {"invite_token": request.token},
                {
                    "$set": {
                        "status": "rejected",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return {
                "success": True,
                "message": "Invitation rejected"
            }
        
        # Accept action
        if not request.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password is required to accept invitation"
            )
        
        if len(request.password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        member_email = member_info["member_email"]
        member_name = member_info["member_name"]
        org_email = member_info["org_email"]
        
        # Check if user already exists with ANY user_type
        existing_user = await users_collection.find_one({"email": member_email})
        
        if existing_user:
            # Check if existing user is already a member of this org
            existing_member = await organization_members_collection.find_one({
                "member_email": member_email,
                "org_email": org_email,
                "status": "accepted"
            })
            
            if existing_member:
                # Already a member, just update status
                await organization_members_collection.update_one(
                    {"invite_token": request.token},
                    {
                        "$set": {
                            "status": "accepted",
                            "accepted_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
            else:
                # User exists but not as member - check if they're organization type
                if existing_user.get("user_type") != "organization":
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="This email is already registered as a candidate. Cannot add as organization member."
                    )
                # User exists as organization type, just update member record
                await organization_members_collection.update_one(
                    {"invite_token": request.token},
                    {
                        "$set": {
                            "status": "accepted",
                            "accepted_at": datetime.utcnow(),
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
        else:
            # Create new user account as organization member
            password_hash = get_password_hash(request.password)
            
            user_doc = {
                "email": member_email,
                "name": member_name,
                "password": password_hash,
                "user_type": "organization",
                "org_member_of": org_email,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            await users_collection.insert_one(user_doc)
            
            # Update member record
            await organization_members_collection.update_one(
                {"invite_token": request.token},
                {
                    "$set": {
                        "status": "accepted",
                        "password_hash": password_hash,
                        "accepted_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        print(f"✅ [POST /api/organization/respond-invite] Member {member_email} accepted invitation")
        
        return {
            "success": True,
            "message": "Invitation accepted successfully. You can now login.",
            "email": member_email
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error responding to invite: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error responding to invite: {str(e)}"
        )


@app.get("/api/organization/members")
async def get_organization_members(current_user: dict = Depends(get_current_user)):
    """
    Get all members of the organization (Owner only)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Check if user is owner
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can view members"
            )
        
        org_email = current_user["email"]
        
        # Get all members
        members_cursor = organization_members_collection.find({
            "org_email": org_email
        }).sort("created_at", -1)
        
        members = []
        async for member in members_cursor:
            member["_id"] = str(member["_id"])
            # Don't send password hash
            if "password_hash" in member:
                del member["password_hash"]
            members.append(member)
        
        return {
            "success": True,
            "members": members
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching members: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching members: {str(e)}"
        )


class ResendInviteRequest(BaseModel):
    member_email: EmailStr

@app.post("/api/organization/resend-invite")
async def resend_invitation(
    request: ResendInviteRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Resend invitation email to a pending member (Owner only)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Check if user is owner
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can resend invitations"
            )
        
        org_email = current_user["email"]
        
        # Find the member
        member_info = await organization_members_collection.find_one({
            "org_email": org_email,
            "member_email": request.member_email
        })
        
        if not member_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Member not found"
            )
        
        if member_info.get("status") == "accepted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Member has already accepted the invitation"
            )
        
        # Generate new invite token
        invite_token = str(uuid.uuid4())
        
        # Update member record with new token
        await organization_members_collection.update_one(
            {"org_email": org_email, "member_email": request.member_email},
            {
                "$set": {
                    "invite_token": invite_token,
                    "invited_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Get organization name
        org_data = await organization_data_collection.find_one({"email": org_email})
        org_name = org_data.get("name", org_email) if org_data else org_email
        
        # Send invitation email in background
        from services.email_service import send_member_invitation_email
        frontend_url = get_frontend_base_url()
        invite_link = f"{frontend_url}/invite-accept/{invite_token}"
        
        background_tasks.add_task(
            send_member_invitation_email,
            request.member_email,
            member_info["member_name"],
            org_name,
            invite_link
        )
        
        print(f"✅ [POST /api/organization/resend-invite] Resent invitation to {request.member_email}")
        
        return {
            "success": True,
            "message": "Invitation resent successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error resending invitation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error resending invitation: {str(e)}"
        )


# ==================== PAYMENT ENDPOINTS ====================

@app.post("/api/payments/buy-credits")
async def buy_credits(
    request: BuyCreditsRequest,
    current_user: dict = Depends(get_current_user)
):
    """Create payment order for buying credits (Org owner only). Supports PayPal and Razorpay."""
    print(f"[BUY_CREDITS] >>> Request hit this backend (localhost:5555 if you see this in local terminal)")
    try:
        # Only org owners can buy credits
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can buy credits"
            )
        
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can purchase credits"
            )
        
        if request.num_credits < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Number of credits must be at least 1"
            )
        
        # Calculate amount (1 credit = $10 USD)
        amount_usd = request.num_credits * 0.011
        org_email = get_org_email(current_user)
        frontend_url = get_frontend_base_url()
        payment_method = getattr(request, "payment_method", None) or "paypal"
        print(f"[BUY_CREDITS] payment_method={payment_method!r} num_credits={request.num_credits}")

        if payment_method == "razorpay":
            # Razorpay: amount in INR (paise). USD to INR rate from env.
            usd_to_inr = float(os.getenv("USD_TO_INR_RATE", "91.93"))
            amount_inr = amount_usd * usd_to_inr
            amount_paise = int(round(amount_inr * 100))
            receipt_id = f"credits_{uuid.uuid4().hex[:12]}"
            razorpay_order = create_razorpay_order(
                amount_paise=amount_paise,
                currency="INR",
                receipt=receipt_id,
                notes={"num_credits": request.num_credits, "org_email": org_email},
            )
            order_id_razor = razorpay_order["id"]
            payment_doc = {
                "order_id": order_id_razor,
                "payment_provider": "razorpay",
                "razorpay_order_id": order_id_razor,
                "org_email": org_email,
                "amount": amount_inr,
                "amount_paise": amount_paise,
                "currency": "INR",
                "num_credits": request.num_credits,
                "status": "created",
                "description": f"Purchase {request.num_credits} credit(s)",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }
            await payments_collection.insert_one(payment_doc)
            razorpay_key_id = os.getenv("RAZORPAY_KEY_ID")
            if not razorpay_key_id:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Razorpay is not configured (RAZORPAY_KEY_ID missing)"
                )
            return {
                "success": True,
                "payment_method": "razorpay",
                "order_id": order_id_razor,
                "razorpay_order_id": order_id_razor,
                "razorpay_key_id": razorpay_key_id,
                "amount_paise": amount_paise,
                "currency": "INR",
                "amount": amount_inr,
                "num_credits": request.num_credits,
            }
        else:
            # PayPal
            result = await create_payment_order(
                amount=amount_usd,
                currency="USD",
                description=f"Purchase {request.num_credits} credit(s) for job postings",
                return_url=f"{frontend_url}/payment/success",
                cancel_url=f"{frontend_url}/payment/cancel"
            )
            payment_doc = {
                "order_id": result["order_id"],
                "payment_provider": "paypal",
                "paypal_order_id": result["paypal_order_id"],
                "org_email": org_email,
                "amount": amount_usd,
                "currency": "USD",
                "num_credits": request.num_credits,
                "status": "created",
                "description": f"Purchase {request.num_credits} credit(s)",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            await payments_collection.insert_one(payment_doc)
            return {
                "success": True,
                "payment_method": "paypal",
                "order_id": result["order_id"],
                "approval_url": result["approval_url"],
                "amount": amount_usd,
                "num_credits": request.num_credits
            }
    
    except HTTPException:
        raise
    except ValueError as e:
        if "RAZORPAY" in str(e):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Razorpay is not configured"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        err_msg = str(e)
        print(f"❌ Error creating credit purchase order: {e}")
        # Surface Razorpay/API errors so frontend can show them (e.g. invalid key)
        if "razorpay" in err_msg.lower() or "invalid" in err_msg.lower() or "unauthorized" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=err_msg
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating payment order: {err_msg}"
        )

@app.post("/api/payments/capture-order")
async def capture_order_endpoint(
    request: CaptureOrderRequest,
    current_user: dict = Depends(get_current_user)
):
    """Capture payment and add credits to organization. Supports PayPal (capture) and Razorpay (verify)."""
    try:
        # Find payment record
        payment = await payments_collection.find_one({
            "order_id": request.order_id,
            "org_email": request.org_email
        })
        
        if not payment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Payment record not found"
            )
        
        if payment.get("status") == "completed":
            # Get current credits
            org_owner = await users_collection.find_one({"email": request.org_email})
            current_credits = org_owner.get("credits", 0) if org_owner else 0
            return {
                "success": True,
                "message": "Payment already completed",
                "credits_added": payment.get("num_credits", 0),
                "total_credits": current_credits
            }
        
        payment_provider = payment.get("payment_provider", "paypal")
        num_credits = payment.get("num_credits", 0)

        if payment_provider == "razorpay":
            # Razorpay: verify signature then mark completed and add credits
            if not request.payment_id or not request.signature:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Razorpay payment requires payment_id and signature for verification"
                )
            if not verify_razorpay_signature(
                order_id=request.order_id,
                payment_id=request.payment_id,
                signature=request.signature,
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Razorpay signature verification failed"
                )
            await payments_collection.update_one(
                {"order_id": request.order_id},
                {
                    "$set": {
                        "status": "completed",
                        "razorpay_payment_id": request.payment_id,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        else:
            # PayPal: capture order
            capture_result = await capture_payment_order(request.order_id)
            if capture_result.get("status") != "completed":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Payment capture failed: {capture_result.get('status')}"
                )
            await payments_collection.update_one(
                {"order_id": request.order_id},
                {
                    "$set": {
                        "status": "completed",
                        "paypal_capture_id": capture_result.get("capture_id"),
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Add credits to organization owner's account
        org_owner = await users_collection.find_one({"email": request.org_email})
        if org_owner:
            current_credits = org_owner.get("credits", 0)
            new_credits = current_credits + num_credits
            await users_collection.update_one(
                {"email": request.org_email},
                {
                    "$set": {
                        "credits": new_credits,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization owner not found"
            )
        
        return {
            "success": True,
            "message": "Payment captured and credits added successfully",
            "credits_added": num_credits,
            "total_credits": new_credits
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error capturing payment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error capturing payment: {str(e)}"
        )

@app.get("/api/organization/credits")
async def get_organization_credits(
    current_user: dict = Depends(get_current_user)
):
    """Get organization's current credits (visible to all org members)"""
    try:
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        org_email = get_org_email(current_user)
        org_owner = await users_collection.find_one({"email": org_email})
        
        if not org_owner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        credits = org_owner.get("credits", 0)
        
        return {
            "success": True,
            "credits": credits,
            "org_email": org_email
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching credits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.get("/api/organization/payment-history")
async def get_payment_history(
    current_user: dict = Depends(get_current_user)
):
    """Get payment transaction history for organization (owner only)"""
    try:
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Only organization owners can view payment history
        if not is_org_owner(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organization owners can view payment history"
            )
        
        org_email = get_org_email(current_user)
        
        # Debug: Log what we're searching for
        print(f"🔍 [PAYMENT HISTORY] Searching for payments with org_email: {org_email}")
        
        # Get all payments for this organization (including all statuses)
        # Try matching both org_email and email fields to catch all payments
        query = {
            "$or": [
                {"org_email": org_email},
                {"email": org_email}  # Fallback for old payment records
            ]
        }
        
        payments_cursor = payments_collection.find(query).sort("created_at", -1)  # Most recent first
        
        transactions = []
        async for payment in payments_cursor:
            # Convert ObjectId to string
            payment["_id"] = str(payment["_id"])
            
            # Format the transaction data
            transaction = {
                "id": str(payment["_id"]),
                "order_id": payment.get("order_id"),
                "amount": payment.get("amount", 0),
                "currency": payment.get("currency", "USD"),
                "num_credits": payment.get("num_credits", 0),
                "description": payment.get("description", ""),
                "status": payment.get("status", "unknown"),
                "created_at": payment.get("created_at"),
                "updated_at": payment.get("updated_at"),
                "paypal_capture_id": payment.get("paypal_capture_id")
            }
            transactions.append(transaction)
        
        # Debug: Log what we found
        print(f"🔍 [PAYMENT HISTORY] Found {len(transactions)} payments for org_email: {org_email}")
        if transactions:
            print(f"   Statuses: {[t['status'] for t in transactions]}")
        
        return {
            "success": True,
            "transactions": transactions,
            "total_transactions": len(transactions)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching payment history: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ==================== ORGANIZATION JOB POSTING ENDPOINTS ====================

class JobPostData(BaseModel):
    role: str
    location: str
    number_of_openings: int
    application_close_date: str
    job_package_lpa: float
    job_type: str  # "full_time", "internship", "unpaid"
    notes: Optional[str] = ""

@app.post("/api/organization-jobpost")
async def create_job_post(
    role: str = Form(...),
    location: str = Form(...),
    number_of_openings: int = Form(...),
    application_close_date: str = Form(...),
    job_package_lpa_min: Optional[float] = Form(None),
    job_package_lpa_max: Optional[float] = Form(None),
    job_package_lpa: Optional[float] = Form(None),  # legacy single value; used if min/max not provided
    job_type: str = Form(...),
    notes: Optional[str] = Form(""),
    jd_file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Create job posting with file upload (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # Get organization email (owner's email)
        org_email = get_org_email(current_user)
        
        # Check credits before posting
        org_owner = await users_collection.find_one({"email": org_email})
        if not org_owner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization not found"
            )
        
        current_credits = org_owner.get("credits", 0)
        if current_credits < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient credits. Please purchase credits to post a job."
            )

        # Get organization data for company info
        org_data = await organization_data_collection.find_one({
            "email": org_email
        })

        if not org_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization profile not found. Please complete your profile first."
            )

        # Check if organization has employees (required for AI notes feature)
        employees_details = org_data.get("employees_details", [])
        if not employees_details or len(employees_details) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please add at least one employee to your organization profile before posting a job. This is required for AI-powered candidate comparison and notes."
            )

        # Handle file upload (S3 only)
        file_path = ""
        if jd_file and jd_file.filename:
            file_ext = os.path.splitext(jd_file.filename)[1] or ".pdf"
            content = await jd_file.read()
            try:
                file_path = upload_bytes_to_s3(
                    content,
                    key_prefix="jds",
                    file_extension=file_ext,
                    content_type=jd_file.content_type or "application/pdf",
                )
            except ValueError as e:
                print(f"❌ S3 not configured: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="File storage (S3) is not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME and AWS_REGION in .env."
                )

        # Resolve package: range (min/max) or legacy single value
        if job_package_lpa_min is not None and job_package_lpa_max is not None:
            pkg_min, pkg_max = float(job_package_lpa_min), float(job_package_lpa_max)
            if job_type != "unpaid" and (pkg_min < 0 or pkg_max < 0):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Job package must be non-negative for paid positions."
                )
            if pkg_min > pkg_max:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Min LPA cannot be greater than Max LPA."
                )
            if job_type == "unpaid" and (pkg_min != 0 or pkg_max != 0):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Job package must be 0 for unpaid positions."
                )
            job_package_lpa_min_val, job_package_lpa_max_val = pkg_min, pkg_max
            job_package_lpa_val = pkg_max  # for backward compat
        elif job_package_lpa is not None:
            pkg = float(job_package_lpa)
            if job_type != "unpaid" and pkg < 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Job package must be non-negative for paid positions."
                )
            if job_type == "unpaid" and pkg != 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Job package must be 0 for unpaid positions."
                )
            job_package_lpa_min_val = job_package_lpa_max_val = pkg
            job_package_lpa_val = pkg
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Provide either job_package_lpa_min and job_package_lpa_max, or job_package_lpa."
            )

        # Generate job ID
        job_id = f"job_{uuid.uuid4()}"

        # Create job document
        job_doc = {
            "job_id": job_id,
            "company": {
                "name": org_data.get("name", current_user.get("name", "")),
                "email": org_email
            },
            "created_by": current_user["email"],  # Track who created the job
            "role": role,
            "file_path": file_path,
            "location": location,
            "number_of_openings": number_of_openings,
            "application_close_date": application_close_date,
            "job_package_lpa_min": job_package_lpa_min_val,
            "job_package_lpa_max": job_package_lpa_max_val,
            "job_package_lpa": job_package_lpa_val,  # backward compat
            "job_type": job_type,
            "notes": notes,
            "applied_candidates": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Save to open-jobs collection
        result = await open_jobs_collection.insert_one(job_doc)

        # Deduct 1 credit after successful job creation
        new_credits = current_credits - 1
        await users_collection.update_one(
            {"email": org_email},
            {
                "$set": {
                    "credits": new_credits,
                    "updated_at": datetime.utcnow()
                }
            }
        )

        print(f"✅ [POST /api/organization-jobpost] Job created: {job_id}, Credits remaining: {new_credits}")

        return {
            "success": True,
            "message": "Job posting created successfully",
            "job_id": job_id,
            "credits_remaining": new_credits
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating job post: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating job post: {str(e)}"
        )


@app.get("/api/organization-jobpost")
async def get_organization_jobs(current_user: dict = Depends(get_current_user)):
    """
    Get organization's job postings (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # First, automatically move any expired jobs to ongoing
        await move_expired_jobs_to_ongoing()

        # Build query based on user role
        org_email = get_org_email(current_user)
        query = {"company.email": org_email}
        
        # If member, filter to only their jobs
        if is_org_member(current_user):
            query["created_by"] = current_user["email"]

        # Get open jobs for this organization
        jobs_cursor = open_jobs_collection.find(query).sort("created_at", -1)

        jobs = []
        async for job in jobs_cursor:
            # Convert ObjectId to string
            job["_id"] = str(job["_id"])

            # For each job, get the current status of applicants from job-applied collection
            if job.get("applied_candidates"):
                for candidate in job["applied_candidates"]:
                    # Get status from job-applied collection
                    applied_record = await job_applied_collection.find_one({
                        "job_id": job["job_id"],
                        "email": candidate["email"]
                    })
                    if applied_record:
                        candidate["status"] = applied_record.get("status", "applied")
                        candidate["applied_at"] = applied_record.get("applied_at")
                    else:
                        candidate["status"] = "applied"

            jobs.append(job)

        return {
            "success": True,
            "jobs": jobs,
            "user_role": current_user.get("role", "owner")  # Include role in response
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching job posts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/api/organization-jobpost/{job_id}/close")
async def close_job_posting(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Move job from ongoing-jobs to closed-jobs collection (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        org_email = get_org_email(current_user)
        
        # Find the job in ongoing-jobs collection
        job = await ongoing_jobs_collection.find_one({
            "job_id": job_id,
            "company.email": org_email
        })

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or not in ongoing status"
            )
        
        # Check permission
        if not can_edit_job(job, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to close this job"
            )

        # Convert ObjectId to string for transfer
        job["_id"] = str(job["_id"])
        job["closed_at"] = datetime.utcnow()

        # Move to closed-jobs collection
        await closed_jobs_collection.insert_one(job)

        # Remove from ongoing-jobs collection
        await ongoing_jobs_collection.delete_one({"job_id": job_id})

        print(f"✅ [PUT /api/organization-jobpost/{job_id}/close] Job closed: {job_id}")

        return {
            "success": True,
            "message": "Job posting closed successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error closing job post: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error closing job post: {str(e)}"
        )


@app.put("/api/organization-jobpost/{job_id}/applicant/{candidate_email}/status")
async def update_applicant_status(
    job_id: str,
    candidate_email: str,
    status_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Update applicant status in job-applied collection and job's applied_candidates array (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        new_status = status_data.get("status")
        valid_statuses = ["applied", "decision_pending", "decision_pending_review", "selected_for_interview", "rejected", "selected", "processing", "invitation_sent", "offer_sent", "offer_accepted"]
        if not new_status or new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )

        org_email = get_org_email(current_user)
        
        # Find the job to check permissions
        job = None
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job = await collection.find_one({
                "job_id": job_id,
                "company.email": org_email
            })
            if job:
                break
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Check permission
        if not can_edit_job(job, current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to update applicants for this job"
            )

        # Update status in job-applied collection
        result_applied = await job_applied_collection.update_one(
            {"job_id": job_id, "email": candidate_email},
            {"$set": {"status": new_status, "updated_at": datetime.utcnow()}}
        )

        # Update status in the job's applied_candidates array
        # Check all three collections (open, ongoing, closed)
        collections = [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]

        updated = False
        for collection in collections:
            result_job = await collection.update_one(
                {
                    "job_id": job_id,
                    "company.email": org_email,
                    "applied_candidates.email": candidate_email
                },
                {
                    "$set": {
                        "applied_candidates.$.status": new_status,
                        "applied_candidates.$.updated_at": datetime.utcnow()
                    }
                }
            )

            if result_job.modified_count > 0:
                updated = True
                break

        if result_applied.modified_count == 0 and not updated:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Applicant not found in this job"
            )

        print(f"✅ [PUT /api/organization-jobpost/{job_id}/applicant/{candidate_email}/status] Status updated to: {new_status}")

        return {
            "success": True,
            "message": "Applicant status updated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error updating applicant status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating applicant status: {str(e)}"
        )


@app.get("/api/organization-jobpost/ongoing")
async def get_ongoing_job_posts(current_user: dict = Depends(get_current_user)):
    """
    Get organization's ongoing job postings (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # Build query based on user role
        org_email = get_org_email(current_user)
        query = {"company.email": org_email}
        
        # If member, filter to only their jobs
        if is_org_member(current_user):
            query["created_by"] = current_user["email"]

        # Get ongoing jobs for this organization
        jobs_cursor = ongoing_jobs_collection.find(query).sort("created_at", -1)

        jobs = []
        async for job in jobs_cursor:
            # Convert ObjectId to string
            job["_id"] = str(job["_id"])

            # For each job, get the current status of applicants from job-applied collection
            if job.get("applied_candidates"):
                for candidate in job["applied_candidates"]:
                    # Get status from job-applied collection
                    applied_record = await job_applied_collection.find_one({
                        "job_id": job["job_id"],
                        "email": candidate["email"]
                    })
                    if applied_record:
                        candidate["status"] = applied_record.get("status", "applied")
                        candidate["applied_at"] = applied_record.get("applied_at")
                    else:
                        candidate["status"] = "applied"

            jobs.append(job)

        return {
            "success": True,
            "jobs": jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching ongoing job posts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/organization-jobpost/{job_id}/applicants")
async def get_job_applicants(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get all applicants for a specific job with their details from job-applied collection (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # Get all applicants for this job from job-applied collection
        applicants_cursor = job_applied_collection.find({
            "job_id": job_id
        }).sort("applied_at", -1)

        applicants = []
        async for applicant in applicants_cursor:
            # Convert ObjectId to string
            applicant["_id"] = str(applicant["_id"])
            
            # Get user profile data if available
            user_profile = await user_data_collection.find_one({
                "user_email": applicant.get("email", "")
            })
            
            if user_profile:
                applicant["profile"] = {
                    "name": user_profile.get("name", applicant.get("name", "")),
                    "email": user_profile.get("user_email", applicant.get("email", "")),
                    "location": user_profile.get("location", ""),
                    "phone": user_profile.get("phone", ""),
                    "gender": user_profile.get("gender", ""),
                    "dob": user_profile.get("dob", ""),
                    "resume_url": user_profile.get("resumeUrl", user_profile.get("resume_url", "")),
                    "parsed_resume_data": user_profile.get("parsed_resume_data"),
                    "additional_details": applicant.get("additional_details", "")
                }
            
            # If status is invitation_sent, fetch webhook/invitation details
            if applicant.get("status") == "invitation_sent":
                # Get the most recent webhook for this candidate (not cancelled)
                webhook_cursor = interview_webhook_collection.find({
                    "job_id": job_id,
                    "applicantEmail": applicant.get("email"),
                    "status": {"$ne": "cancelled"}  # Exclude cancelled invitations
                }).sort("created_at", -1).limit(1)
                
                webhook_data = None
                async for webhook in webhook_cursor:
                    webhook_data = webhook
                    break
                
                # If no active webhook found, try to get any webhook (including cancelled ones)
                if not webhook_data:
                    webhook_cursor = interview_webhook_collection.find({
                        "job_id": job_id,
                        "applicantEmail": applicant.get("email")
                    }).sort("created_at", -1).limit(1)
                    
                    async for webhook in webhook_cursor:
                        webhook_data = webhook
                        break
                
                if webhook_data:
                    # Add invitation details to applicant (will be displayed in ongoing_rounds format)
                    if not applicant.get("ongoing_rounds"):
                        applicant["ongoing_rounds"] = []
                    
                    # Create a round object from webhook data
                    invitation_round = {
                        "round": webhook_data.get("round"),
                        "team": webhook_data.get("team"),
                        "location_type": webhook_data.get("location_type", "online"),
                        "status": "invitation_sent",
                        "webhook_id": webhook_data.get("webhook_id"),
                        "sent_at": webhook_data.get("created_at")
                    }
                    
                    # If already scheduled (status=submitted in webhook), add scheduling details
                    if webhook_data.get("status") == "submitted":
                        invitation_round.update({
                            "interviewer_name": webhook_data.get("interviewer_name"),
                            "interviewer_email": webhook_data.get("interviewer_email"),
                            "interview_date": webhook_data.get("selected_date"),
                            "interview_time": webhook_data.get("selected_time"),
                            "meeting_link": webhook_data.get("meeting_link"),
                            "location": webhook_data.get("location"),
                            "scheduled_at": webhook_data.get("submitted_at"),
                            "status": "scheduled"
                        })
                    
                    applicant["ongoing_rounds"].append(invitation_round)
                else:
                    # Log for debugging - webhook not found but status is invitation_sent
                    print(f"⚠️ Warning: Candidate {applicant.get('email')} has status 'invitation_sent' but no webhook found for job_id {job_id}")
            
            # If status is offer_sent, offer_accepted, or rejected (after offer), attach offer letter and responded_at from offer webhook
            if applicant.get("status") in ("offer_sent", "offer_accepted", "rejected"):
                offer_cursor = offer_webhook_collection.find({
                    "job_id": job_id,
                    "applicantEmail": applicant.get("email", "")
                }).sort("created_at", -1).limit(1)
                offer_webhook = None
                async for ow in offer_cursor:
                    offer_webhook = ow
                    break
                if offer_webhook:
                    applicant["offer_letter_path"] = offer_webhook.get("offer_letter_path")
                    applicant["offer_responded_at"] = offer_webhook.get("responded_at")
            
            applicants.append(applicant)

        return {
            "success": True,
            "applicants": applicants
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching job applicants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/organization-jobpost/closed")
async def get_closed_job_posts(current_user: dict = Depends(get_current_user)):
    """
    Get organization's closed job postings (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # Build query based on user role
        org_email = get_org_email(current_user)
        query = {"company.email": org_email}
        
        # If member, filter to only their jobs
        if is_org_member(current_user):
            query["created_by"] = current_user["email"]

        # Get closed jobs for this organization
        jobs_cursor = closed_jobs_collection.find(query).sort("closed_at", -1)

        jobs = []
        async for job in jobs_cursor:
            # Convert ObjectId to string
            job["_id"] = str(job["_id"])

            # For each job, get the current status of applicants from job-applied collection
            if job.get("applied_candidates"):
                for candidate in job["applied_candidates"]:
                    # Get status from job-applied collection
                    applied_record = await job_applied_collection.find_one({
                        "job_id": job["job_id"],
                        "email": candidate["email"]
                    })
                    if applied_record:
                        candidate["status"] = applied_record.get("status", "applied")
                        candidate["applied_at"] = applied_record.get("applied_at")
                    else:
                        candidate["status"] = "applied"

            # Count offer_accepted applicants AFTER updating statuses
            # This ensures the count matches the actual candidates in applied_candidates array
            offer_accepted_count = 0
            if job.get("applied_candidates"):
                offer_accepted_count = sum(1 for candidate in job["applied_candidates"] 
                                          if candidate.get("status") == "offer_accepted")
            
            # Add offer_accepted_count to job object
            job["offer_accepted_count"] = offer_accepted_count

            jobs.append(job)

        return {
            "success": True,
            "jobs": jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching closed job posts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== JOB STATUS MANAGEMENT ====================

async def move_expired_jobs_to_ongoing():
    """
    Background task to move jobs from open-jobs to ongoing-jobs when application deadline has passed
    """
    try:
        # Get today's date in YYYY-MM-DD format
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # Get all expired jobs (where application_close_date <= today)
        expired_jobs_cursor = open_jobs_collection.find({
            "application_close_date": {"$lte": today}
        })

        moved_count = 0
        async for job in expired_jobs_cursor:
            # Convert ObjectId to string for transfer
            job["_id"] = str(job["_id"])
            job["moved_to_ongoing_at"] = datetime.utcnow()

            # Move to ongoing-jobs collection
            await ongoing_jobs_collection.insert_one(job)

            # Remove from open-jobs collection
            await open_jobs_collection.delete_one({"job_id": job["job_id"]})

            moved_count += 1

        if moved_count > 0:
            print(f"✅ Moved {moved_count} expired jobs to ongoing-jobs collection")

        return moved_count

    except Exception as e:
        print(f"❌ Error moving expired jobs: {e}")
        return 0


@app.post("/api/admin/manage-job-status")
async def manage_job_status():
    """
    Admin endpoint to move expired jobs to ongoing (can be called periodically)
    """
    try:
        moved_count = await move_expired_jobs_to_ongoing()
        return {
            "success": True,
            "message": f"Moved {moved_count} jobs to ongoing status"
        }
    except Exception as e:
        print(f"❌ Error in manage job status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.put("/api/organization-jobpost/{job_id}/apply")
async def apply_for_job(
    job_id: str,
    application_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Apply for a job posting (add candidate to applied_candidates array)
    """
    try:
        # Verify user is a candidate (regular user)
        if current_user.get("user_type") != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only candidates can apply for jobs"
            )

        # Check if user has already applied (job-applied is source of truth)
        applied_exists = await job_applied_collection.find_one({
            "job_id": job_id,
            "email": current_user["email"]
        })
        if applied_exists:
            print(f"⚠️ [APPLY] Already registered: {current_user['email']} for job {job_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Already registered"
            )

        # Get job details (must exist in open jobs to apply)
        job_details = await open_jobs_collection.find_one({"job_id": job_id})
        if not job_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        # Get user profile for resume URL
        user_profile = await user_data_collection.find_one({"user_email": current_user["email"]})
        resume_url = application_data.get("resume_url", "")
        if not resume_url and user_profile:
            resume_url = user_profile.get("resumeUrl", "") or user_profile.get("resume_url", "")
        print(f"📄 [APPLY] Resume URL for {current_user['email']}: {resume_url}")

        user_additional_details = (application_data.get("additional_details") or "").strip()
        if not resume_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume not found. Please upload your resume in your profile before applying."
            )

        # Require phone number from user profile (not from resume)
        profile_phone = (user_profile or {}).get("phone") or ""
        if not profile_phone or str(profile_phone).strip() == "":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Please add your phone number in your profile before applying."
            )

        # Get resume bytes from S3 only (no local storage)
        resume_bytes = None
        if is_s3_url(resume_url):
            resume_bytes = download_from_s3_url(resume_url)
        if not resume_bytes:
            print(f"⚠️ [APPLY] Resume file not found: url={resume_url}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Your resume file could not be found. Please re-upload your resume in your profile so we can run the evaluation."
            )

        # Parse resume with OpenAI and save to user profile (support PDF/DOCX via RAG)
        parser = ResumeParser()
        try:
            resume_text = ""
            try:
                from services.rag import extract_text_from_file
                resume_text = extract_text_from_file(resume_bytes, "resume.pdf")
                if resume_text and resume_text.startswith("Error:"):
                    resume_text = ""
            except ImportError:
                pass
            if not resume_text or len(resume_text.strip()) < 10:
                resume_text = parser.extract_text_from_pdf(resume_bytes)
            if not resume_text or len(resume_text.strip()) < 10:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Could not extract text from your resume. Please ensure the PDF contains readable text."
                )
            parsed_resume_data = parser.parse(resume_text)
        except ValueError as e:
            print(f"❌ [APPLY] Resume parse error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not parse your resume. Please ensure the PDF contains readable text."
            )
        except Exception as e:
            print(f"❌ [APPLY] Resume parse error: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Your resume could not be processed. Please re-upload a valid PDF."
            )

        await user_data_collection.update_one(
            {"user_email": current_user["email"]},
            {"$set": {"parsed_resume_data": parsed_resume_data, "updated_at": datetime.utcnow()}},
            upsert=True
        )

        # JD text for comparator (S3 or local)
        jd_text = _read_text_from_static_file_or_s3(static_dir, job_details.get("file_path", ""))
        if not jd_text:
            jd_text = f"""Job Role: {job_details.get('role', 'Info not given')}
Company: {job_details.get('company', {}).get('name', 'Info not given')}
Location: {job_details.get('location', 'Info not given')}
Notes: {job_details.get('notes', '')}"""

        # Build employees_data from org for previous-employee comparison
        org_email = job_details.get("company", {}).get("email")
        org_data = await organization_data_collection.find_one({"email": org_email}) if org_email else None
        employees_data = _build_employees_data_from_org(static_dir, org_data)

        # Run comparator agent (AI evaluation) with previous employees
        additional_info = {}
        if user_additional_details:
            additional_info["additional_details"] = user_additional_details
        try:
            comparator = ComparatorAgent(employees_data=employees_data)
            ai_evaluation = comparator.process(
                candidate_data=parsed_resume_data,
                job_description=jd_text,
                additional_info=additional_info
            )
        except Exception as e:
            print(f"⚠️ [APPLY] ComparatorAgent error (continuing without AI summary): {e}")
            ai_evaluation = user_additional_details or ""

        combined_details = (user_additional_details + "\n\n--- AI Evaluation ---\n" + ai_evaluation).strip() if user_additional_details else ai_evaluation

        candidate_data = {
            "name": application_data.get("name", current_user.get("name", "")),
            "email": current_user["email"],
            "resume_url": resume_url,
            "additional_details": combined_details or ""
        }
        
        # Add candidate to applied_candidates array in open-jobs collection
        result = await open_jobs_collection.update_one(
            {"job_id": job_id},
            {"$push": {"applied_candidates": candidate_data}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or already closed"
            )

        # Also save to job-applied collection for user's tracking
        applied_job_data = {
            "job_id": job_id,
            "name": candidate_data["name"],
            "email": current_user["email"],
            "resume_url": resume_url,
            "status": "applied",
            "additional_details": combined_details,
            "applied_at": datetime.utcnow()
        }

        await job_applied_collection.insert_one(applied_job_data)

        print(f"✅ [PUT /api/organization-jobpost/{job_id}/apply] Application submitted by: {current_user['email']}")

        return {
            "success": True,
            "message": "Job applied"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error applying for job: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying for job: {str(e)}"
        )


@app.get("/api/jobs")
async def get_all_open_jobs(current_user: dict = Depends(get_current_user)):
    """
    Get all open job postings for candidates to view (Protected - users only)
    """
    try:
        # Verify user is a candidate (regular user)
        if current_user.get("user_type") != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only candidates can view job listings"
            )

        # First, automatically move any expired jobs to ongoing
        await move_expired_jobs_to_ongoing()

        # Get all open jobs
        jobs_cursor = open_jobs_collection.find({}).sort("created_at", -1)

        jobs = []
        async for job in jobs_cursor:
            # Convert ObjectId to string
            job["_id"] = str(job["_id"])
            jobs.append(job)

        return {
            "success": True,
            "jobs": jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching jobs for candidates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/jobs/applied")
async def get_user_applied_jobs(current_user: dict = Depends(get_current_user)):
    """
    Get user's applied jobs with full job details (Protected - users only)
    """
    try:
        # Verify user is a candidate (regular user)
        if current_user.get("user_type") != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only candidates can view applied jobs"
            )

        # Get user's applied jobs from job-applied collection
        applied_cursor = job_applied_collection.find({
            "email": current_user["email"]
        }).sort("applied_at", -1)

        applied_jobs = []
        async for applied_job in applied_cursor:
            job_id = applied_job["job_id"]
            job_details = None
            
            # IMPORTANT: Check all three collections (open, ongoing, closed) for job details
            # This ensures applied jobs are shown regardless of current job status
            collections = [
                ("open", open_jobs_collection),
                ("ongoing", ongoing_jobs_collection),
                ("closed", closed_jobs_collection)
            ]
            
            for collection_name, collection in collections:
                job_details = await collection.find_one({"job_id": job_id})
                if job_details:
                    print(f"✅ Found job {job_id} in {collection_name} collection")
                    break

            if job_details:
                # Convert ObjectId to string
                job_details["_id"] = str(job_details["_id"])

                # Combine applied job data with full job details
                combined_job = {
                    **job_details,
                    "application_status": applied_job.get("status", "applied"),
                    "applied_at": applied_job.get("applied_at"),
                    "additional_details": applied_job.get("additional_details", "")
                }
                applied_jobs.append(combined_job)
            else:
                # If job not found in any collection, still include it with basic info
                # This ensures ALL applied jobs are shown regardless of job status or existence
                print(f"⚠️ Job {job_id} not found in any collection, but including in applied jobs")
                basic_job = {
                    "job_id": job_id,
                    "role": "Job details unavailable",
                    "company": {
                        "name": "Unknown",
                        "email": ""
                    },
                    "location": "Info not given",
                    "application_status": applied_job.get("status", "applied"),
                    "applied_at": applied_job.get("applied_at"),
                    "additional_details": applied_job.get("additional_details", ""),
                    "job_status": "unknown"
                }
                applied_jobs.append(basic_job)
        
        print(f"✅ Returning {len(applied_jobs)} applied jobs for user {current_user['email']}")

        return {
            "success": True,
            "jobs": applied_jobs
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching applied jobs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/public/jobs/{job_id}")
async def get_public_job(job_id: str):
    """
    Get a single job by ID for public viewing (no auth required).
    Used for shareable job links. Does not return applied_candidates.
    """
    try:
        await move_expired_jobs_to_ongoing()
        job = await open_jobs_collection.find_one({"job_id": job_id})
        if job is not None:
            job["job_status"] = "open"
        else:
            job = await ongoing_jobs_collection.find_one({"job_id": job_id})
            if job is not None:
                job["job_status"] = "ongoing"
            else:
                job = await closed_jobs_collection.find_one({"job_id": job_id})
                if job is not None:
                    job["job_status"] = "closed"
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        # Return a copy without applied_candidates (do not expose applicant data)
        job = dict(job)
        job["_id"] = str(job["_id"])
        job.pop("applied_candidates", None)
        return {"success": True, "job": job}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching public job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/jobs/{job_id}")
async def get_single_job(
    job_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get a single job by ID for candidates (Protected - users only).
    Looks in open, then ongoing, then closed so users can view details
    for jobs they applied to or that have moved past open.
    """
    try:
        if current_user.get("user_type") != "user":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only candidates can view job details"
            )
        await move_expired_jobs_to_ongoing()
        job = await open_jobs_collection.find_one({"job_id": job_id})
        if job is not None:
            job["job_status"] = "open"
        else:
            job = await ongoing_jobs_collection.find_one({"job_id": job_id})
            if job is not None:
                job["job_status"] = "ongoing"
            else:
                job = await closed_jobs_collection.find_one({"job_id": job_id})
                if job is not None:
                    job["job_status"] = "closed"
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        job["_id"] = str(job["_id"])
        # Check if current user has already applied (for candidates)
        user_has_applied = False
        applied_doc = await job_applied_collection.find_one({
            "job_id": job_id,
            "email": current_user.get("email")
        })
        if applied_doc:
            user_has_applied = True
        return {"success": True, "job": job, "user_has_applied": user_has_applied}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== DEMO REQUEST ENDPOINTS ====================

@app.post("/api/demo-request")
async def create_demo_request(request: DemoRequest):
    """
    Handle demo request submission and send email notification
    """
    try:
        # Send email notification
        email_sent = send_demo_request_email(
            name=request.name,
            email=request.email,
            phone=request.phone,
            company_name=request.companyName,
            position=request.position,
            date=request.date,
            time=request.time,
            comments=request.comments or ""
        )
        
        if email_sent:
            return {
                "success": True,
                "message": "Demo request submitted successfully",
                "data": {
                    "name": request.name,
                    "email": request.email,
                    "companyName": request.companyName
                }
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send email notification"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing demo request: {str(e)}"
        )


# ==================== ORGANIZATION TEAM ENDPOINTS ====================

@app.post("/api/organization-teams")
async def create_or_update_team(
    team_data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Create or update organization team data (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        # Remove _id if it exists
        if "_id" in team_data:
            team_data.pop("_id")

        org_email = get_org_email(current_user)
        
        # Add/update organization email, creator, and timestamp
        team_data["organization_email"] = org_email
        team_data["created_by"] = current_user["email"]  # Track creator
        team_data["updated_at"] = datetime.utcnow()

        # Create or update the teams document for this organization in separate collection
        # Filter by creator for members, by org_email for owner
        if is_org_member(current_user):
            query = {"organization_email": org_email, "created_by": current_user["email"]}
        else:
            query = {"organization_email": org_email}
        
        result = await organization_teams_collection.update_one(
            query,
            {"$set": team_data},
            upsert=True
        )

        print(f"✅ [POST /api/organization-teams] Update result: {result.modified_count} modified, {result.upserted_id} upserted")

        return {
            "success": True,
            "message": "Team data saved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving team data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving team data: {str(e)}"
        )


@app.get("/api/organization-teams")
async def get_organization_teams(current_user: dict = Depends(get_current_user)):
    """
    Get organization's team data (Protected)
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )

        org_email = get_org_email(current_user)
        
        # Build query
        if is_org_member(current_user):
            # Members see only their teams
            query = {"organization_email": org_email, "created_by": current_user["email"]}
        else:
            # Owners see all teams
            query = {"organization_email": org_email}
        
        # Get organization teams data from separate collection
        teams_data = await organization_teams_collection.find_one(query)

        if not teams_data:
            # Return empty teams structure if not found
            return {
                "success": True,
                "teams": []
            }

        # Convert ObjectId to string
        teams_data["_id"] = str(teams_data["_id"])

        # Return teams data or empty array
        return {
            "success": True,
            "teams": teams_data.get("teams", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching team data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ==================== INTERVIEW SCHEDULING ENDPOINTS ====================

class ScheduleInterviewRequest(BaseModel):
    applicantEmail: str
    applicantName: str
    orgEmail: str
    orgName: str
    round: str
    team: str

class FreeSlotsRequest(BaseModel):
    orgEmail: str
    orgName: str
    teamName: str
    webhook_id: Optional[str] = None  # Optional for validation

@app.post("/api/get-free-slots")
async def get_free_slots(request: FreeSlotsRequest):
    """
    Get free time slots for a team based on team members' calendars
    Public endpoint - no authentication required (for applicant form)
    """
    try:
        # Optional: Validate webhook_id if provided
        if request.webhook_id:
            webhook_record = await interview_webhook_collection.find_one({
                "webhook_id": request.webhook_id
            })
            if not webhook_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Invalid webhook ID"
                )
            # Verify the webhook matches the request
            if webhook_record.get("orgEmail") != request.orgEmail or webhook_record.get("team") != request.teamName:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Webhook data mismatch"
                )
        
        # Fetch team data
        teams_data = await organization_teams_collection.find_one({
            "organization_email": request.orgEmail
        })
        
        if not teams_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team data not found"
            )
        
        # Find the specific team
        teams = teams_data.get("teams", [])
        target_team = None
        for team in teams:
            if team.get("team_name") == request.teamName:
                target_team = team
                break
        
        if not target_team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Team '{request.teamName}' not found"
            )
        
        # Get calendar links from team members
        members = target_team.get("members", [])
        calendar_links = []
        for member in members:
            calendar_link = member.get("calendar_link", "")
            if calendar_link:
                calendar_links.append(calendar_link)
        
        if not calendar_links:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No calendar links found for team members"
            )
        
        # Get free slots from calendars
        free_slots = get_group_free_slots(calendar_links, days=5)
        
        # Get all booked slots for this organization and team
        # Find all submitted interviews for the same orgEmail and team
        booked_slots = await interview_webhook_collection.find({
            "orgEmail": request.orgEmail,
            "team": request.teamName,
            "status": "submitted",
            "selected_slot_id": {"$exists": True}
        }).to_list(length=1000)
        
        # Create a set of booked slot IDs for quick lookup
        booked_slot_ids = set()
        for booked in booked_slots:
            slot_id = booked.get("selected_slot_id")
            if slot_id:
                booked_slot_ids.add(slot_id)
        
        # Filter out booked slots from free slots
        available_slots = {}
        for date_key, slots_list in free_slots.items():
            available_slots_for_date = [
                slot for slot in slots_list
                if slot.get("slot_id") not in booked_slot_ids
            ]
            # Only add date if there are available slots
            if available_slots_for_date:
                available_slots[date_key] = available_slots_for_date
        
        # If no slots available, send notification email to organization
        no_slots = len(available_slots) == 0
        if no_slots:
            from services.email_service import send_no_slots_notification_email
            try:
                await send_no_slots_notification_email(
                    org_email=request.orgEmail,
                    org_name=request.orgName,
                    team_name=request.teamName
                )
            except Exception as e:
                print(f"⚠️ Error sending no slots notification: {e}")
        
        return {
            "success": True,
            "free_slots": available_slots,
            "no_slots_available": no_slots
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error getting free slots: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting free slots: {str(e)}"
        )


class SendInterviewFormRequest(BaseModel):
    applicantEmail: str
    applicantName: str
    orgEmail: str
    orgName: str
    round: str
    team: str
    job_id: str
    location_type: str  # "online" or "offline"
    is_ai_interview: Optional[bool] = False  # NEW: Flag for AI interview

@app.post("/api/send-interview-form")
async def send_interview_form(
    request: SendInterviewFormRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Send interview scheduling form email to applicant
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Get frontend URL from environment (shareable links: AI interview, schedule-interview)
        frontend_url = get_frontend_base_url()
        # Check candidate's current status
        applicant_doc = await job_applied_collection.find_one({
            "job_id": request.job_id,
            "email": request.applicantEmail
        })
        
        candidate_status = applicant_doc.get("status") if applicant_doc else None
        
        # If scheduling an AI interview: same job, same person — no two AI interviews
        is_ai_interview = getattr(request, 'is_ai_interview', False)
        if is_ai_interview:
            existing_ai = await interview_feedback_collection.find_one({
                "job_id": request.job_id,
                "applicant_email": request.applicantEmail,
                "type": "ai_interview",
                "status": "completed"
            })
            if existing_ai:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="This candidate has already completed an AI interview for this job. Same job, same person — you cannot schedule another AI interview."
                )
        
        # Check if there's a pending invitation (not yet scheduled) for the same round
        # Allow rescheduling same round if candidate is in "selected_for_interview" (Conduct Rounds)
        # This allows conducting the same round multiple times
        existing_pending_invitation = await interview_webhook_collection.find_one({
            "applicantEmail": request.applicantEmail,
            "job_id": request.job_id,
            "round": request.round,
            "status": {"$ne": "submitted"}  # Only block if invitation sent but not yet scheduled
        })
        
        # If candidate is in "selected_for_interview" status, allow scheduling same round again
        # This enables conducting the same round multiple times
        if existing_pending_invitation and candidate_status != "selected_for_interview":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"An interview invitation for {request.round} has already been sent to this applicant. Please wait for them to schedule before sending another invitation for this round."
            )
        
        # If there's a pending invitation but candidate is in "selected_for_interview", 
        # cancel the old invitation to allow new one
        if existing_pending_invitation and candidate_status == "selected_for_interview":
            await interview_webhook_collection.update_one(
                {"webhook_id": existing_pending_invitation.get("webhook_id")},
                {
                    "$set": {
                        "status": "cancelled",
                        "cancelled_at": datetime.utcnow(),
                        "reason": "Replaced by new invitation for same round"
                    }
                }
            )
            print(f"ℹ️ Cancelling old invitation to allow rescheduling same round for {request.applicantEmail}")
        
        # Create unique webhook ID for this interview request
        webhook_id = str(uuid.uuid4())
        
        # Create form link or AI interview link
        if is_ai_interview:
            # AI interview direct link
            form_link = f"{frontend_url}/ai-interview?job_id={request.job_id}&email={request.applicantEmail}"
        else:
            # Regular interview scheduling form
            form_link = f"{frontend_url}/schedule-interview?webhook_id={webhook_id}&applicantEmail={request.applicantEmail}&applicantName={request.applicantName}&orgEmail={request.orgEmail}&orgName={request.orgName}&round={request.round}&team={request.team}"
        
        # Store webhook mapping first to ensure correct matching
        await interview_webhook_collection.insert_one({
            "webhook_id": webhook_id,
            "applicantEmail": request.applicantEmail,
            "applicantName": request.applicantName,
            "orgEmail": request.orgEmail,
            "orgName": request.orgName,
            "round": request.round,
            "team": request.team if not is_ai_interview else "",
            "job_id": request.job_id,
            "location_type": request.location_type if not is_ai_interview else "ai_online",
            "is_ai_interview": is_ai_interview,
            "interview_link": form_link if is_ai_interview else "",
            "created_at": datetime.utcnow()
        })
        
        # Update job-applied collection: set status to invitation_sent
        await job_applied_collection.update_one(
            {"job_id": request.job_id, "email": request.applicantEmail},
            {
                "$set": {
                    "status": "invitation_sent",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update status in job collections
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            await collection.update_one(
                {
                    "job_id": request.job_id,
                    "applied_candidates.email": request.applicantEmail
                },
                {
                    "$set": {
                        "applied_candidates.$.status": "invitation_sent",
                        "applied_candidates.$.updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Send email using email service (after status update to ensure it happens)
        try:
            if is_ai_interview:
                from services.email_service import send_ai_interview_invitation_email
                jd_file_path = None
                job = None
                for coll in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                    job = await coll.find_one({"job_id": request.job_id})
                    if job:
                        break
                if job:
                    fp = (job.get("file_path") or "").strip()
                    if fp:
                        jd_file_path = _local_path_or_download_s3(static_dir, fp)  # S3 only
                try:
                    success = await send_ai_interview_invitation_email(
                    applicant_email=request.applicantEmail,
                    applicant_name=request.applicantName,
                    interview_link=form_link,
                    round_name=request.round,
                    org_name=request.orgName,
                    job_id=request.job_id,
                    jd_file_path=jd_file_path
                )
                finally:
                    if jd_file_path and job and is_s3_url((job.get("file_path") or "")) and str(jd_file_path).startswith(tempfile.gettempdir() or "/tmp"):
                        try:
                            os.unlink(jd_file_path)
                        except Exception:
                            pass
            else:
                # Regular interview scheduling email
                success = await send_interview_form_email(
                    applicant_email=request.applicantEmail,
                    applicant_name=request.applicantName,
                    form_link=form_link,
                    round_name=request.round,
                    org_name=request.orgName
                )
            
            if not success:
                print(f"⚠️ Warning: Email sending failed for {request.applicantEmail}, but status has been updated")
        except Exception as e:
            print(f"❌ Error sending interview form email: {e}")
            print(f"⚠️ Warning: Email sending failed for {request.applicantEmail}, but status has been updated")
            success = False
        
        # Return success even if email fails (status is updated, user can resend if needed)
        return {
            "success": True,
            "message": "Interview invitation sent successfully" if success else "Interview invitation created, but email sending failed. Please check email configuration.",
            "webhook_id": webhook_id,
            "email_sent": success
        }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error sending interview form: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending interview form: {str(e)}"
        )


class CheckWebhookStatusRequest(BaseModel):
    webhook_id: str

@app.post("/api/check-webhook-status")
async def check_webhook_status(request: CheckWebhookStatusRequest):
    """
    Check if webhook has already been submitted
    """
    try:
        webhook_record = await interview_webhook_collection.find_one({
            "webhook_id": request.webhook_id
        })
        
        if not webhook_record:
            return {
                "success": True,
                "exists": False,
                "submitted": False
            }
        
        is_submitted = webhook_record.get("status") == "submitted"
        
        return {
            "success": True,
            "exists": True,
            "submitted": is_submitted,
            "data": {
                "selected_date": webhook_record.get("selected_date"),
                "selected_time": webhook_record.get("selected_time"),
                "location_type": webhook_record.get("location_type"),
                "location": webhook_record.get("location"),
                "submitted_at": webhook_record.get("submitted_at")
            } if is_submitted else None
        }
    except Exception as e:
        print(f"❌ Error checking webhook status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking webhook status: {str(e)}"
        )


class SubmitInterviewFormRequest(BaseModel):
    webhook_id: str
    selected_date: str
    selected_slot_id: str
    selected_time: str  # Format: "08:00 AM - 08:30 AM"

@app.post("/api/submit-interview-form")
async def submit_interview_form(request: SubmitInterviewFormRequest):
    """
    Receive interview form submission from applicant
    """
    try:
        # Find the webhook record
        webhook_record = await interview_webhook_collection.find_one({
            "webhook_id": request.webhook_id
        })
        
        if not webhook_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid webhook ID"
            )
        
        # Check if already submitted
        if webhook_record.get("status") == "submitted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This interview has already been scheduled"
            )
        
        # Import required services
        from services.calendar_service import find_free_team_members_at_time, create_calendar_invite
        from services.meet_service import create_meet_space
        from services.email_service import send_interview_confirmation_emails
        import pytz
        from dateutil import parser
        
        ist = pytz.timezone("Asia/Kolkata")
        
        # Parse selected date and time
        date_str = request.selected_date
        time_str = request.selected_time.split(" - ")[0]  # Get start time
        
        # Parse date and time - handle format like "Wednesday 24-Dec-2025" and "05:00 PM"
        datetime_str = f"{date_str} {time_str}"
        try:
            selected_datetime = parser.parse(datetime_str)
            if selected_datetime.tzinfo is None:
                selected_datetime = ist.localize(selected_datetime)
            else:
                selected_datetime = selected_datetime.astimezone(ist)
        except Exception as e:
            # Fallback parsing
            print(f"⚠️ Error parsing datetime, trying alternative: {e}")
            # Try to parse slot_id format: 20251224_1700_1730
            slot_parts = request.selected_slot_id.split("_")
            if len(slot_parts) >= 3:
                date_part = slot_parts[0]  # 20251224
                time_part = slot_parts[1]  # 1700
                year = int(date_part[:4])
                month = int(date_part[4:6])
                day = int(date_part[6:8])
                hour = int(time_part[:2])
                minute = int(time_part[2:4])
                selected_datetime = ist.localize(datetime(year, month, day, hour, minute))
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Could not parse selected date/time: {datetime_str}"
                )
        
        end_datetime = selected_datetime + timedelta(minutes=30)
        
        # Check if slot is still available (double-check before booking)
        # Get all booked slots for this organization and team
        booked_slots_check = await interview_webhook_collection.find({
            "orgEmail": webhook_record.get("orgEmail"),
            "team": webhook_record.get("team"),
            "status": "submitted",
            "selected_slot_id": request.selected_slot_id
        }).to_list(length=10)
        
        if booked_slots_check:
            # Slot is already booked by someone else
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This time slot has been booked by another applicant. Please refresh and select a different time."
            )
        
        # Get team members
        teams_data = await organization_teams_collection.find_one({
            "organization_email": webhook_record.get("orgEmail")
        })
        
        if not teams_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team data not found"
            )
        
        teams = teams_data.get("teams", [])
        target_team = None
        for team in teams:
            if team.get("team_name") == webhook_record.get("team"):
                target_team = team
                break
        
        if not target_team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Team not found"
            )
        
        # Find free team members at selected time
        free_members = find_free_team_members_at_time(
            target_team.get("members", []),
            selected_datetime,
            duration_minutes=30
        )
        
        if not free_members:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No team members are available at the selected time"
            )
        
        # Select only ONE interviewer (randomly from available members to distribute load)
        selected_interviewer = random.choice(free_members)
        interviewer_email = selected_interviewer.get("email")
        interviewer_name = selected_interviewer.get("name", interviewer_email)
        
        if not interviewer_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Selected interviewer does not have an email address"
            )
        
        # Get location_type from webhook record (set by organization when sending invitation)
        location_type = webhook_record.get("location_type", "online")
        location = None
        meeting_link = None
        
        # Always create a meeting link (even for offline interviews - for backup/virtual option)
        try:
            meeting_link = create_meet_space()
        except Exception as e:
            print(f"⚠️ Error creating Meet space: {e}")
            meeting_link = f"https://meet.google.com/new"
        
        # Get job_id early (needed for location fetch)
        job_id = webhook_record.get("job_id")
        
        if location_type == "offline":
            # Fetch location from ongoing-jobs based on job_id
            location = None
            
            if job_id:
                # First check ongoing-jobs collection (as requested)
                job_details_for_location = await ongoing_jobs_collection.find_one({"job_id": job_id})
                
                # If not found in ongoing, check other collections as fallback
                if not job_details_for_location:
                    for collection in [open_jobs_collection, closed_jobs_collection]:
                        job_details_for_location = await collection.find_one({"job_id": job_id})
                        if job_details_for_location:
                            break
                
                if job_details_for_location:
                    location = job_details_for_location.get("location", "")
                    if not location:
                        print(f"⚠️ Job location not found for job_id: {job_id}")
                else:
                    print(f"⚠️ Job not found for job_id: {job_id}")
            else:
                print(f"⚠️ job_id not found in webhook record")
            
            # For offline: meeting_link is still created (as backup), but location is primary
        else:
            # Online: meeting_link is primary
            location_type = "online"
        
        # Get job details and JD file (S3 only: download to temp for email attachment)
        job_details = None
        jd_file_path = None
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job_details = await collection.find_one({"job_id": job_id})
            if job_details:
                jd_file_path_str = job_details.get("file_path", "")
                if jd_file_path_str:
                    jd_file_path = _local_path_or_download_s3(static_dir, jd_file_path_str)
                break

        # Get applicant resume (S3 only: download to temp for email attachment)
        applicant_email = webhook_record.get("applicantEmail")
        user_profile = await user_data_collection.find_one({"user_email": applicant_email})
        resume_file_path = None
        resume_url = ""
        if user_profile:
            resume_url = user_profile.get("resumeUrl") or user_profile.get("resume_url", "") or ""
            if resume_url:
                resume_file_path = _local_path_or_download_s3(static_dir, resume_url)
        
        # Prepare attendee emails - only applicant and one interviewer
        attendee_emails = [applicant_email, interviewer_email]
        interviewer_emails = [interviewer_email]  # Only one interviewer
        
        # Create calendar invite
        organizer_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
        if organizer_email:
            ics_content = create_calendar_invite(
                meeting_link=meeting_link,
                start_datetime=selected_datetime,
                end_datetime=end_datetime,
                summary=f"Interview - {webhook_record.get('round')}",
                description=f"Interview with {webhook_record.get('applicantName')} for {webhook_record.get('round')}",
                organizer_email=organizer_email,
                attendee_emails=attendee_emails
            )
            
            # Send calendar invites via email
            password = os.getenv("EMAIL_PASSWORD")
            
            if password:
                for email_addr in attendee_emails:
                    try:
                        msg = MIMEMultipart()
                        msg["From"] = organizer_email
                        msg["To"] = email_addr
                        msg["Subject"] = f"Interview Invitation - {webhook_record.get('round')}"
                        msg.attach(MIMEText(f"Please find the calendar invite attached."))
                        
                        part = MIMEText(ics_content, "calendar;method=REQUEST")
                        part.add_header("Content-Disposition", "attachment; filename=invite.ics")
                        msg.attach(part)
                        
                        with smtplib.SMTP("smtp.gmail.com", 587) as server:
                            server.starttls()
                            server.login(organizer_email, password)
                            server.sendmail(organizer_email, email_addr, msg.as_string())
                        print(f"✅ Calendar invite sent to {email_addr}")
                    except Exception as e:
                        print(f"⚠️ Error sending calendar invite to {email_addr}: {e}")
        
        # Create feedback form webhook ID first (needed for email)
        feedback_webhook_id = str(uuid.uuid4())
        frontend_url = get_frontend_base_url()
        feedback_form_link = f"{frontend_url}/interview-feedback?feedback_id={feedback_webhook_id}&webhook_id={request.webhook_id}"
        
        # Store feedback webhook mapping
        await interview_feedback_collection.insert_one({
            "feedback_id": feedback_webhook_id,
            "webhook_id": request.webhook_id,
            "job_id": job_id,
            "applicant_email": applicant_email,
            "applicant_name": webhook_record.get("applicantName"),
            "interviewer_email": interviewer_email,
            "interviewer_name": interviewer_name,
            "round": webhook_record.get("round"),
            "interview_date": request.selected_date,
            "interview_time": request.selected_time,
            "meeting_link": meeting_link,
            "location_type": location_type,
            "location": location,
            "status": "pending",  # pending, submitted
            "created_at": datetime.utcnow()
        })
        
        # Previous rounds + AI transcript (S3 URL) for interviewer email
        previous_rounds_html, is_first_round, transcript_url = await _get_previous_rounds_and_transcript(
            job_id, applicant_email, static_dir
        )

        # Send confirmation emails (combined - includes feedback form link for interviewer)
        try:
            await send_interview_confirmation_emails(
                applicant_email=applicant_email,
                applicant_name=webhook_record.get("applicantName"),
                interviewer_emails=interviewer_emails,
                interview_date=request.selected_date,
                interview_time=request.selected_time,
                meeting_link=meeting_link,
                round_name=webhook_record.get("round"),
                org_name=webhook_record.get("orgName"),
                job_id=job_id,
                location_type=location_type,
                location=location,
                feedback_form_link=feedback_form_link,
                jd_file_path=str(jd_file_path) if jd_file_path else None,
                resume_file_path=str(resume_file_path) if resume_file_path else None,
                previous_rounds_section=previous_rounds_html,
                is_first_round=is_first_round,
                transcript_url=transcript_url,
            )
        finally:
            # Clean up temp files if we downloaded from S3
            tmp_prefix = tempfile.gettempdir() or "/tmp"
            if jd_file_path and job_details and is_s3_url((job_details.get("file_path") or "")) and str(jd_file_path).startswith(tmp_prefix):
                try:
                    os.unlink(jd_file_path)
                except Exception:
                    pass
            if resume_file_path and resume_url and is_s3_url(resume_url) and str(resume_file_path).startswith(tmp_prefix):
                try:
                    os.unlink(resume_file_path)
                except Exception:
                    pass
        
        # Log the submission (print to console as requested)
        print("\n" + "="*80)
        print("📅 INTERVIEW SCHEDULING FORM SUBMISSION")
        print("="*80)
        print(f"Applicant Name: {webhook_record.get('applicantName')}")
        print(f"Applicant Email: {applicant_email}")
        print(f"Organization Name: {webhook_record.get('orgName')}")
        print(f"Organization Email: {webhook_record.get('orgEmail')}")
        print(f"Round: {webhook_record.get('round')}")
        print(f"Team: {webhook_record.get('team')}")
        print(f"Job ID: {job_id}")
        print(f"Selected Date: {request.selected_date}")
        print(f"Selected Time: {request.selected_time}")
        print(f"Slot ID: {request.selected_slot_id}")
        print(f"Meeting Link: {meeting_link}")
        print(f"Selected Interviewer: {interviewer_name} ({interviewer_email})")
        print(f"Total Free Interviewers Available: {len(free_members)}")
        print(f"Webhook ID: {request.webhook_id}")
        print("="*80 + "\n")
        
        # Update webhook record with submission
        await interview_webhook_collection.update_one(
            {"webhook_id": request.webhook_id},
            {
                "$set": {
                    "selected_date": request.selected_date,
                    "selected_time": request.selected_time,
                    "selected_slot_id": request.selected_slot_id,
                    "meeting_link": meeting_link,
                    "location_type": location_type,
                    "location": location,
                    "interviewer_email": interviewer_email,
                    "interviewer_name": interviewer_name,
                    "selected_datetime": selected_datetime.isoformat(),
                    "submitted_at": datetime.utcnow(),
                    "status": "submitted"
                }
            }
        )
        
        # Update job-applied collection: add to ongoing_rounds and update status
        await job_applied_collection.update_one(
            {"job_id": job_id, "email": applicant_email},
            {
                "$set": {
                    "status": "processing",  # Move to ongoing rounds
                    "updated_at": datetime.utcnow()
                },
                "$push": {
                    "ongoing_rounds": {
                        "round": webhook_record.get("round"),
                        "interviewer_email": interviewer_email,
                        "interviewer_name": interviewer_name,
                        "interview_date": request.selected_date,
                        "interview_time": request.selected_time,
                        "meeting_link": meeting_link,
                        "location_type": location_type,
                        "location": location,
                        "scheduled_at": datetime.utcnow(),
                        "feedback_id": feedback_webhook_id,
                        "status": "scheduled"
                    }
                }
            }
        )
        
        # Update status in job collections (from invitation_sent to processing)
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            await collection.update_one(
                {
                    "job_id": job_id,
                    "applied_candidates.email": applicant_email
                },
                {
                    "$set": {
                        "applied_candidates.$.status": "processing",
                        "applied_candidates.$.updated_at": datetime.utcnow()
                    }
                }
            )
        
        return {
            "success": True,
            "message": "Interview scheduled successfully",
            "data": {
                "applicantName": webhook_record.get('applicantName'),
                "applicantEmail": applicant_email,
                "orgName": webhook_record.get('orgName'),
                "orgEmail": webhook_record.get('orgEmail'),
                "round": webhook_record.get('round'),
                "team": webhook_record.get('team'),
                "selected_date": request.selected_date,
                "selected_time": request.selected_time,
                "meeting_link": meeting_link,
                "location_type": location_type,
                "location": location,
                "interviewer": interviewer_name,
                "interviewer_email": interviewer_email
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting interview form: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting interview form: {str(e)}"
        )


class InterviewFeedbackRequest(BaseModel):
    feedback_id: str
    candidate_attended: str  # "yes", "no", "reschedule"
    technical_configuration: int  # 1-5
    technical_customization: int  # 1-5
    communication_skills: int  # 1-5
    leadership_abilities: int  # 1-5
    enthusiasm: int  # 1-5
    teamwork: int  # 1-5
    attitude: int  # 1-5
    comments: str = ""  # Optional comments
    interview_outcome: str  # "selected", "proceed", "rejected"

@app.post("/api/submit-interview-feedback")
async def submit_interview_feedback(request: InterviewFeedbackRequest):
    """
    Receive interview feedback form submission from interviewer
    """
    try:
        # Find the feedback record
        feedback_record = await interview_feedback_collection.find_one({
            "feedback_id": request.feedback_id
        })
        
        if not feedback_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid feedback ID"
            )
        
        # Check if already submitted
        if feedback_record.get("status") == "submitted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Feedback has already been submitted for this interview"
            )
        
        job_id = feedback_record.get("job_id")
        applicant_email = feedback_record.get("applicant_email")
        round_name = feedback_record.get("round")
        
        # Prepare round details
        round_details = {
            "round": round_name,
            "interviewer_email": feedback_record.get("interviewer_email"),
            "interviewer_name": feedback_record.get("interviewer_name"),
            "interview_date": feedback_record.get("interview_date"),
            "interview_time": feedback_record.get("interview_time"),
            "feedback_submitted_at": datetime.utcnow(),
            "candidate_attended": request.candidate_attended
        }
        
        # Handle based on candidate_attended
        if request.candidate_attended == "no":
            # Candidate didn't attend - reject
            round_details["interview_outcome"] = "rejected"
            round_details["reason"] = "Candidate did not attend"
            round_details["scores"] = None
            
            # Update job-applied: move to previous_rounds, update status
            await job_applied_collection.update_one(
                {"job_id": job_id, "email": applicant_email},
                {
                    "$set": {"status": "rejected", "updated_at": datetime.utcnow()},
                    "$pull": {"ongoing_rounds": {"feedback_id": request.feedback_id}},
                    "$push": {"previous_rounds": round_details}
                }
            )
            
            # Update status in job collections
            for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                await collection.update_one(
                    {
                        "job_id": job_id,
                        "applied_candidates.email": applicant_email
                    },
                    {
                        "$set": {
                            "applied_candidates.$.status": "rejected",
                            "applied_candidates.$.updated_at": datetime.utcnow()
                        }
                    }
                )
        
        elif request.candidate_attended == "reschedule":
            # Reschedule - no scores, just note
            round_details["interview_outcome"] = "reschedule"
            round_details["reason"] = "Interview needs to be rescheduled"
            round_details["scores"] = {
                "technical_configuration": 0,
                "technical_customization": 0,
                "communication_skills": 0,
                "leadership_abilities": 0,
                "enthusiasm": 0,
                "teamwork": 0,
                "attitude": 0
            }
            
            # Update job-applied: move to previous_rounds, change status to selected_for_interview (so they can be rescheduled)
            # First, get the current document to find the exact ongoing_round to remove
            applicant_doc = await job_applied_collection.find_one(
                {"job_id": job_id, "email": applicant_email}
            )
            
            if applicant_doc and applicant_doc.get("ongoing_rounds"):
                # Find and remove the specific ongoing round by feedback_id
                ongoing_rounds = applicant_doc.get("ongoing_rounds", [])
                updated_ongoing_rounds = [
                    round_item for round_item in ongoing_rounds 
                    if round_item.get("feedback_id") != request.feedback_id
                ]
                
                await job_applied_collection.update_one(
                    {"job_id": job_id, "email": applicant_email},
                    {
                        "$set": {
                            "status": "selected_for_interview",
                            "updated_at": datetime.utcnow(),
                            "ongoing_rounds": updated_ongoing_rounds
                        },
                        "$push": {"previous_rounds": round_details}
                    }
                )
            else:
                # Fallback: use $pull if document structure is different
                await job_applied_collection.update_one(
                    {"job_id": job_id, "email": applicant_email},
                    {
                        "$set": {"status": "selected_for_interview", "updated_at": datetime.utcnow()},
                        "$pull": {"ongoing_rounds": {"feedback_id": request.feedback_id}},
                        "$push": {"previous_rounds": round_details}
                    }
                )
            
            # Update status in job collections
            for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                await collection.update_one(
                    {
                        "job_id": job_id,
                        "applied_candidates.email": applicant_email
                    },
                    {
                        "$set": {
                            "applied_candidates.$.status": "selected_for_interview",
                            "applied_candidates.$.updated_at": datetime.utcnow()
                        }
                    }
                )
        
        else:  # candidate_attended == "yes"
            # Candidate attended - process scores and outcome
            scores = {
                "technical_configuration": request.technical_configuration,
                "technical_customization": request.technical_customization,
                "communication_skills": request.communication_skills,
                "leadership_abilities": request.leadership_abilities,
                "enthusiasm": request.enthusiasm,
                "teamwork": request.teamwork,
                "attitude": request.attitude
            }
            
            round_details["scores"] = scores
            round_details["interview_outcome"] = request.interview_outcome
            if request.comments:
                round_details["comments"] = request.comments
            
            if request.interview_outcome == "rejected":
                # Rejected - move to previous_rounds, update status
                await job_applied_collection.update_one(
                    {"job_id": job_id, "email": applicant_email},
                    {
                        "$set": {"status": "rejected", "updated_at": datetime.utcnow()},
                        "$pull": {"ongoing_rounds": {"feedback_id": request.feedback_id}},
                        "$push": {"previous_rounds": round_details}
                    }
                )
                
                # Update status in job collections
                for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                    await collection.update_one(
                        {
                            "job_id": job_id,
                            "applied_candidates.email": applicant_email
                        },
                        {
                            "$set": {
                                "applied_candidates.$.status": "rejected",
                                "applied_candidates.$.updated_at": datetime.utcnow()
                            }
                        }
                    )
            
            elif request.interview_outcome == "proceed":
                # Proceed to next round - move to selected_for_interview status
                await job_applied_collection.update_one(
                    {"job_id": job_id, "email": applicant_email},
                    {
                        "$set": {"status": "selected_for_interview", "updated_at": datetime.utcnow()},
                        "$pull": {"ongoing_rounds": {"feedback_id": request.feedback_id}},
                        "$push": {"previous_rounds": round_details}
                    }
                )
                
                # Update status in job collections
                for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                    await collection.update_one(
                        {
                            "job_id": job_id,
                            "applied_candidates.email": applicant_email
                        },
                        {
                            "$set": {
                                "applied_candidates.$.status": "selected_for_interview",
                                "applied_candidates.$.updated_at": datetime.utcnow()
                            }
                        }
                    )
            
            elif request.interview_outcome == "selected":
                # Selected - final selection
                await job_applied_collection.update_one(
                    {"job_id": job_id, "email": applicant_email},
                    {
                        "$set": {"status": "selected", "updated_at": datetime.utcnow()},
                        "$pull": {"ongoing_rounds": {"feedback_id": request.feedback_id}},
                        "$push": {"previous_rounds": round_details}
                    }
                )
                
                # Update status in job collections
                for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
                    await collection.update_one(
                        {
                            "job_id": job_id,
                            "applied_candidates.email": applicant_email
                        },
                        {
                            "$set": {
                                "applied_candidates.$.status": "selected",
                                "applied_candidates.$.updated_at": datetime.utcnow()
                            }
                        }
                    )
        
        # Mark feedback as submitted
        await interview_feedback_collection.update_one(
            {"feedback_id": request.feedback_id},
            {
                "$set": {
                    "status": "submitted",
                    "submitted_at": datetime.utcnow(),
                    "feedback_data": {
                        "candidate_attended": request.candidate_attended,
                        "technical_configuration": request.technical_configuration,
                        "technical_customization": request.technical_customization,
                        "communication_skills": request.communication_skills,
                        "leadership_abilities": request.leadership_abilities,
                        "enthusiasm": request.enthusiasm,
                        "teamwork": request.teamwork,
                        "attitude": request.attitude,
                        "comments": request.comments or "",
                        "interview_outcome": request.interview_outcome
                    }
                }
            }
        )
        
        return {
            "success": True,
            "message": "Feedback submitted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting feedback: {str(e)}"
        )

@app.post("/api/check-feedback-status")
async def check_feedback_status(request: Dict[str, Any] = Body(...)):
    """
    Check if feedback form has already been submitted
    """
    try:
        feedback_id = request.get("feedback_id")
        if not feedback_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="feedback_id is required"
            )
        
        feedback_record = await interview_feedback_collection.find_one({
            "feedback_id": feedback_id
        })
        
        if not feedback_record:
            return {
                "success": True,
                "exists": False,
                "submitted": False
            }
        
        is_submitted = feedback_record.get("status") == "submitted"
        
        return {
            "success": True,
            "exists": True,
            "submitted": is_submitted
        }
    except Exception as e:
        print(f"❌ Error checking feedback status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking feedback status: {str(e)}"
        )


# ==================== OFFER MANAGEMENT ENDPOINTS ====================

class SendOfferLetterRequest(BaseModel):
    applicantEmail: str
    applicantName: str
    orgEmail: str
    orgName: str
    job_id: str

@app.post("/api/send-offer-letter")
async def send_offer_letter(
    applicantEmail: str = Form(...),
    applicantName: str = Form(...),
    orgEmail: str = Form(...),
    orgName: str = Form(...),
    job_id: str = Form(...),
    offer_letter: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Send offer letter to candidate with accept/reject webhook form
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        # Save offer letter to S3 only
        file_ext = os.path.splitext(offer_letter.filename)[1] or ""
        content = await offer_letter.read()
        try:
            file_path = upload_bytes_to_s3(
                content,
                key_prefix="offers",
                file_extension=file_ext,
                content_type=offer_letter.content_type,
            )
        except ValueError as e:
            print(f"❌ S3 not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="File storage (S3) is not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME and AWS_REGION in .env."
            )
        
        # Create webhook ID for offer response
        webhook_id = str(uuid.uuid4())
        frontend_url = get_frontend_base_url()
        form_link = f"{frontend_url}/offer-response?offer_id={webhook_id}"
        
        # Store offer webhook (store S3 URL or local path)
        await offer_webhook_collection.insert_one({
            "webhook_id": webhook_id,
            "applicantEmail": applicantEmail,
            "applicantName": applicantName,
            "orgEmail": orgEmail,
            "orgName": orgName,
            "job_id": job_id,
            "offer_letter_path": file_path,
            "status": "pending",  # pending, accepted, rejected
            "created_at": datetime.utcnow()
        })
        
        # Update status to offer_sent
        await job_applied_collection.update_one(
            {"job_id": job_id, "email": applicantEmail},
            {
                "$set": {
                    "status": "offer_sent",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update status in job collections
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            await collection.update_one(
                {
                    "job_id": job_id,
                    "applied_candidates.email": applicantEmail
                },
                {
                    "$set": {
                        "applied_candidates.$.status": "offer_sent",
                        "applied_candidates.$.updated_at": datetime.utcnow()
                    }
                }
            )
        
        # Get job role from open, ongoing, or closed jobs
        job_details = await open_jobs_collection.find_one({"job_id": job_id})
        if job_details is None:
            job_details = await ongoing_jobs_collection.find_one({"job_id": job_id})
        if job_details is None:
            job_details = await closed_jobs_collection.find_one({"job_id": job_id})
        job_role = job_details.get("role", "the position") if job_details else "the position"
        
        # Send email with offer letter (S3 only: download to temp for attachment)
        from services.email_service import send_offer_letter_email
        offer_local_path = _local_path_or_download_s3(static_dir, file_path)
        try:
            success = await send_offer_letter_email(
                applicant_email=applicantEmail,
                applicant_name=applicantName,
                org_name=orgName,
                job_role=job_role,
                offer_letter_path=offer_local_path or "",
                form_link=form_link
            )
        finally:
            if offer_local_path and is_s3_url(file_path) and offer_local_path.startswith(tempfile.gettempdir() or "/tmp"):
                try:
                    os.unlink(offer_local_path)
                except Exception:
                    pass
        return {
            "success": True,
            "message": "Offer letter sent successfully" if success else "Offer letter created, but email sending failed",
            "webhook_id": webhook_id,
            "email_sent": success
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error sending offer letter: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error sending offer letter: {str(e)}"
        )


class SubmitOfferResponseRequest(BaseModel):
    offer_id: str
    response: str  # "accept" or "reject"

@app.post("/api/submit-offer-response")
async def submit_offer_response(request: SubmitOfferResponseRequest):
    """
    Handle offer acceptance/rejection from candidate
    """
    try:
        # Find the webhook record
        webhook_record = await offer_webhook_collection.find_one({
            "webhook_id": request.offer_id
        })
        
        if not webhook_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid offer ID"
            )
        
        # Check if already responded
        if webhook_record.get("status") != "pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offer response has already been submitted"
            )
        
        job_id = webhook_record.get("job_id")
        applicant_email = webhook_record.get("applicantEmail")
        
        # Update webhook status
        new_status = "accepted" if request.response == "accept" else "rejected"
        await offer_webhook_collection.update_one(
            {"webhook_id": request.offer_id},
            {
                "$set": {
                    "status": new_status,
                    "responded_at": datetime.utcnow()
                }
            }
        )
        
        # Update candidate status
        candidate_status = "offer_accepted" if request.response == "accept" else "rejected"
        await job_applied_collection.update_one(
            {"job_id": job_id, "email": applicant_email},
            {
                "$set": {
                    "status": candidate_status,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update status in job collections
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            await collection.update_one(
                {
                    "job_id": job_id,
                    "applied_candidates.email": applicant_email
                },
                {
                    "$set": {
                        "applied_candidates.$.status": candidate_status,
                        "applied_candidates.$.updated_at": datetime.utcnow()
                    }
                }
            )
        
        return {
            "success": True,
            "message": f"Offer {new_status} successfully",
            "status": candidate_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting offer response: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error submitting offer response: {str(e)}"
        )


# ==================== UTILITY ENDPOINTS ====================

@app.get("/")
async def root():
    return {
        "message": "PRISM API - Complete Backend",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "auth": {
                "signup": "POST /api/auth/signup",
                "verify_otp": "POST /api/auth/verify-otp",
                "login": "POST /api/auth/login",
                "logout": "POST /api/auth/logout",
                "forgot_password": "POST /api/auth/forgot-password",
                "reset_password": "POST /api/auth/reset-password",
                "refresh_token": "POST /api/auth/refresh-token"
            },
            "profile": {
                "create": "POST /api/user-profile (Protected)",
                "get": "GET /api/user-profile (Protected)"
            },
            "demo": {
                "request": "POST /api/demo-request"
            },
            "upload": {
                "file": "POST /api/upload"
            },
            "job_management": {
                "manage_status": "POST /api/admin/manage-job-status"
            }
        }
    }


@app.get("/health")
async def health_check():
    try:
        # Check DB connection
        await client.admin.command('ping')
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/api/review-request")
async def create_review_request(
    data: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Create a decision review request and email a reviewer.
    """
    try:
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can request reviews"
            )

        job_id = data.get("job_id")
        applicant_email = data.get("applicant_email")
        applicant_name = data.get("applicant_name")
        reviewer_email = data.get("reviewer_email")
        resume_url = data.get("resume_url", "")
        additional_details = data.get("additional_details", "")

        if not (job_id and applicant_email and applicant_name and reviewer_email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required fields"
            )

        review_id = str(uuid.uuid4())

        review_doc = {
            "review_id": review_id,
            "job_id": job_id,
            "applicant_email": applicant_email,
            "applicant_name": applicant_name,
            "reviewer_email": reviewer_email,
            "resume_url": resume_url,
            "additional_details": additional_details,
            "status": "pending",
            "created_at": datetime.utcnow()
        }

        await review_requests_collection.insert_one(review_doc)

        frontend_url = get_frontend_base_url()
        review_link = f"{frontend_url}/review-form?review_id={review_id}"

        # Resume link: S3 -> Object URL only; legacy /static/ -> backend base
        backend_base = os.getenv("BACKEND_URL", "http://localhost:5555")
        normalized_resume_url = (resume_url or "").lstrip("/static/")
        resume_link = (to_object_url(resume_url) if is_s3_url(resume_url) else (f"{backend_base}/static/{normalized_resume_url}" if normalized_resume_url else "")) if resume_url else "Not available"

        # Get resume file path for attachment (S3 only: download to temp)
        resume_file_path = None
        resume_file_name = None
        if resume_url:
            resume_file_path = _local_path_or_download_s3(static_dir, resume_url)
            if resume_file_path:
                file_ext = Path(resume_file_path).suffix if Path(resume_file_path).suffix else ".pdf"
                resume_file_name = f"{applicant_name.replace(' ', '_')}_Resume{file_ext}"
                print(f"✅ Resume file found for attachment: {resume_file_path}")
            else:
                print(f"⚠️ Resume file not found for: {resume_url}")

        # Create professional HTML email template
        from services.email_service import send_email_with_attachment
        from_email = os.getenv("FROM_EMAIL") or os.getenv("FROM_email")
        password = os.getenv("EMAIL_PASSWORD")
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6; 
            color: #111827;
            background: linear-gradient(135deg, #facc15 0%, #eab308 100%);
            margin: 0;
            padding: 40px 20px;
        }}
        .container {{ 
            max-width: 640px; 
            margin: 0 auto; 
            background: #ffffff;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        }}
        .header {{ 
            background: linear-gradient(135deg, #facc15 0%, #eab308 100%);
            color: white; 
            padding: 40px 32px; 
            text-align: center;
        }}
        .header-icon {{
            width: 80px;
            height: 80px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-size: 42px;
            backdrop-filter: blur(10px);
        }}
        .header h2 {{
            margin: 0;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.5px;
        }}
        .header p {{
            margin: 8px 0 0 0;
            font-size: 15px;
            opacity: 0.95;
            font-weight: 500;
        }}
        .content {{ 
            background: #ffffff; 
            padding: 36px 32px;
        }}
        .content p {{
            margin: 0 0 16px 0;
            font-size: 15px;
            line-height: 1.7;
            color: #4b5563;
        }}
        .content p.greeting {{
            font-size: 16px;
            font-weight: 600;
            color: #111827;
            margin-bottom: 20px;
        }}
        .details-box {{
            background: linear-gradient(to bottom right, #fef9c3, #fef08a);
            padding: 24px;
            border-radius: 16px;
            margin: 28px 0;
            border: 2px solid #facc15;
            box-shadow: 0 4px 12px rgba(250, 204, 21, 0.15);
        }}
        .details-box h3 {{
            margin: 0 0 16px 0;
            font-size: 18px;
            font-weight: 700;
            color: #b45309;
        }}
        .details-box p {{
            margin: 10px 0;
            font-size: 14px;
            color: #1f2937;
        }}
        .details-box strong {{
            font-weight: 700;
            color: #111827;
            min-width: 140px;
            display: inline-block;
        }}
        .button {{ 
            display: inline-block; 
            padding: 18px 40px; 
            background: linear-gradient(135deg, #facc15 0%, #eab308 100%);
            color: #1f2937; 
            text-decoration: none; 
            border-radius: 14px; 
            margin: 24px 0;
            font-weight: 700;
            font-size: 16px;
            box-shadow: 0 8px 24px rgba(250, 204, 21, 0.4);
            transition: all 0.3s;
            text-align: center;
            letter-spacing: 0.3px;
        }}
        .button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 28px rgba(250, 204, 21, 0.5);
        }}
        .info-note {{
            background: linear-gradient(to right, #dbeafe, #e0e7ff);
            border-left: 4px solid #3b82f6;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 24px 0;
        }}
        .info-note p {{
            margin: 0;
            color: #1e40af;
            font-weight: 500;
            font-size: 14px;
        }}
        .link-box {{
            background: #f9fafb;
            border: 2px dashed #cbd5e1;
            padding: 16px 20px;
            border-radius: 12px;
            margin: 20px 0;
            word-break: break-all;
        }}
        .link-box p {{
            margin: 0;
            color: #facc15;
            font-size: 13px;
            font-weight: 600;
            text-align: center;
        }}
        .footer {{ 
            text-align: center;
            background: linear-gradient(to bottom, #f9fafb, #f3f4f6);
            padding: 24px 32px;
            border-top: 2px solid #e5e7eb;
        }}
        .footer p {{
            margin: 6px 0;
            color: #6b7280;
            font-size: 13px;
        }}
        a {{
            color: #2563eb;
            text-decoration: none;
            font-weight: 600;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-icon">📋</div>
            <h2>Candidate Review Request</h2>
            <p>Your decision is needed</p>
        </div>
        <div class="content">
            <p class="greeting">👋 Hello,</p>
            <p>You have been asked to review a candidate for a position. Please review the candidate's profile and provide your decision.</p>
            <div class="details-box">
                <h3>📋 Candidate Information</h3>
                <p><strong>👤 Name:</strong> {applicant_name}</p>
                <p><strong>📧 Email:</strong> <a href="mailto:{applicant_email}">{applicant_email}</a></p>
                <p><strong>📄 Resume:</strong> <a href="{resume_link}" target="_blank">View Resume →</a></p>
            </div>
            {f'<div class="details-box" style="background: linear-gradient(to bottom right, #f0f9ff, #e0f2fe); border-color: #3b82f6;"><h3 style="color: #1e40af;">📝 Additional Details</h3><div style="font-size: 13px; line-height: 1.6; color: #475569; white-space: pre-wrap;">{additional_details}</div></div>' if additional_details else ''}
            <div class="info-note">
                <p>📌 Please review the candidate's profile and submit your decision using the button below.</p>
            </div>
            <div style="text-align: center; margin: 32px 0;">
                <a href="{review_link}" class="button">📋 Submit Review Decision</a>
            </div>
            <p style="text-align: center; color: #6b7280; font-size: 14px; margin: 20px 0;">If the button doesn't work, copy and paste this link into your browser:</p>
            <div class="link-box">
                <p>{review_link}</p>
            </div>
            <p style="margin-top: 32px; padding-top: 24px; border-top: 2px solid #e5e7eb;">
                <strong>Best regards,</strong><br>
                <strong style="color: #facc15; font-size: 16px;">PRISM Recruiting Team</strong>
            </p>
        </div>
        <div class="footer">
            <p><strong>© 2024 PRISM</strong> - APEXNEURAL</p>
            <p>This is an automated message. Please do not reply to this email.</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Build plain text email (avoiding backslashes in f-string expressions)
        additional_details_text = ""
        if additional_details:
            additional_details_text = f"\nAdditional Details:\n{additional_details}\n"
        
        plain_text = f"""Candidate Review Request

Hello,

You have been asked to review a candidate for a position.

Candidate Information:
Name: {applicant_name}
Email: {applicant_email}
Resume: {resume_link}
{additional_details_text}Please submit your decision (Selected / Rejected / Interview) here:
{review_link}

Best regards,
PRISM Recruiting Team
"""
        
        if not from_email or not password:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Email service not configured"
            )
        
        # Send email with resume attachment if available
        if resume_file_path and resume_file_name:
            try:
                email_sent = send_email_with_attachment(
                    to_email=reviewer_email,
                    subject=f"Review Request - {applicant_name}",
                    html_content=html_body,
                    attachment_path=resume_file_path,
                    attachment_name=resume_file_name
                )
            finally:
                if resume_url and is_s3_url(resume_url) and resume_file_path and str(resume_file_path).startswith(tempfile.gettempdir() or "/tmp"):
                    try:
                        os.unlink(resume_file_path)
                    except Exception:
                        pass
        else:
            # Fallback to send_mail if no resume file
            from services.email_service import send_mail
            email_sent = send_mail(
                to_emails=reviewer_email,
                subject=f"Review Request - {applicant_name}",
                message=plain_text,
                password=password,
                from_email=from_email,
                html_content=html_body
            )
        
        if not email_sent:
            print(f"⚠️ Warning: Failed to send review request email to {reviewer_email}")
        else:
            print(f"✅ Review request email sent successfully to {reviewer_email}")

        return {"success": True, "review_id": review_id}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error creating review request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/review-request/{review_id}")
async def get_review_request(review_id: str):
    """
    Get review request data by review_id (public endpoint for reviewers)
    """
    try:
        review_doc = await review_requests_collection.find_one({"review_id": review_id})
        
        if not review_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review request not found"
            )
        
        # Check if already submitted
        if review_doc.get("status") != "pending":
            return {
                "success": True,
                "review_id": review_doc.get("review_id"),
                "applicant_name": review_doc.get("applicant_name"),
                "applicant_email": review_doc.get("applicant_email"),
                "resume_url": review_doc.get("resume_url", ""),
                "additional_details": review_doc.get("additional_details", ""),
                "status": review_doc.get("status"),
                "submitted_at": review_doc.get("submitted_at"),
                "already_submitted": True
            }
        
        return {
            "success": True,
            "review_id": review_doc.get("review_id"),
            "applicant_name": review_doc.get("applicant_name"),
            "applicant_email": review_doc.get("applicant_email"),
            "resume_url": review_doc.get("resume_url", ""),
            "additional_details": review_doc.get("additional_details", ""),
            "status": review_doc.get("status"),
            "already_submitted": False
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching review request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/review-form/{review_id}")
async def submit_review_decision(
    review_id: str,
    body: Dict[str, Any] = Body(...)
):
    """
    Submit review decision (Selected / Rejected / Interview)
    Updates applicant status in job-applied and open-jobs collections
    """
    try:
        decision = body.get("decision")
        
        if decision not in ["selected", "rejected", "selected_for_interview"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid decision. Must be one of: selected, rejected, selected_for_interview"
            )
        
        # Find review request
        review_doc = await review_requests_collection.find_one({"review_id": review_id})
        
        if not review_doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Review request not found"
            )
        
        # Check if already submitted
        if review_doc.get("status") != "pending":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Review decision has already been submitted"
            )
        
        job_id = review_doc.get("job_id")
        applicant_email = review_doc.get("applicant_email")
        
        # Update review request status
        await review_requests_collection.update_one(
            {"review_id": review_id},
            {
                "$set": {
                    "status": decision,
                    "submitted_at": datetime.utcnow()
                }
            }
        )
        
        # Update applicant status in job-applied collection
        await job_applied_collection.update_one(
            {"job_id": job_id, "email": applicant_email},
            {"$set": {"status": decision}}
        )
        
        # Update applicant status in open-jobs collection (applied_candidates array)
        await open_jobs_collection.update_one(
            {"job_id": job_id, "applied_candidates.email": applicant_email},
            {"$set": {"applied_candidates.$.status": decision}}
        )
        
        print(f"✅ Review decision submitted: {decision} for {applicant_email} (job: {job_id})")
        
        return {
            "success": True,
            "message": "Review decision submitted successfully",
            "decision": decision
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error submitting review decision: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# AI INTERVIEW ENDPOINTS
# ============================================================================

from services.ai_interview_service import AIInterviewAgent, MongoDBTranscriptStorage
from fastapi import WebSocket, WebSocketDisconnect

@app.post("/api/ai-interview/start")
async def start_ai_interview(
    job_id: str = Body(...),
    email: str = Body(...)
):
    """
    Initialize AI interview session.
    Fetch JD and resume from MongoDB, return session_id and WebSocket URL.
    """
    try:
        # Fetch job details from all job collections
        job = None
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job = await collection.find_one({"job_id": job_id})
            if job:
                break
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Extract JD text from file (supports PDF, DOCX, TXT)
        jd_text = _read_text_from_static_file_or_s3(static_dir, job.get("file_path", ""))
        if not jd_text:
            pkg_min = job.get('job_package_lpa_min')
            pkg_max = job.get('job_package_lpa_max')
            if pkg_min is not None and pkg_max is not None and pkg_min != pkg_max:
                pkg_str = f"{pkg_min} - {pkg_max} LPA"
            else:
                pkg_str = f"{job.get('job_package_lpa', 'Info not given')} LPA"
            jd_text = f"""Job Role: {job.get('role', 'Info not given')}
Location: {job.get('location', 'Info not given')}
Job Type: {job.get('job_type', 'Info not given')}
Package: {pkg_str}
Number of Openings: {job.get('number_of_openings', 'Info not given')}
Notes: {job.get('notes', 'Info not given')}"""
        
        # Fetch applicant details
        applicant = await job_applied_collection.find_one({
            "job_id": job_id,
            "email": email
        })
        
        if not applicant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Applicant not found for this job"
            )
        
        # Extract resume text (supports PDF, DOCX, TXT); fallback to user_data if job_applied has no resume_url
        resume_text = _read_text_from_static_file_or_s3(static_dir, applicant.get("resume_url", ""))
        if not resume_text and applicant.get("email"):
            user_profile = await user_data_collection.find_one({"user_email": applicant["email"]})
            if user_profile:
                resume_text = _read_text_from_static_file_or_s3(
                    static_dir,
                    user_profile.get("resumeUrl") or user_profile.get("resume_url") or ""
                )
        
        # Create session
        session_id = str(uuid.uuid4())
        
        # Check if interview already completed
        existing_completed = await interview_feedback_collection.find_one({
            "applicant_email": email,
            "job_id": job_id,
            "type": "ai_interview",
            "status": "completed"
        })
        
        if existing_completed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Interview already completed. This link can only be used once."
            )
        
        # Get webhook record to find round name
        webhook = await interview_webhook_collection.find_one({
            "applicantEmail": email,
            "job_id": job_id,
            "is_ai_interview": True,
            "status": {"$ne": "completed"}
        })
        
        round_name = webhook.get("round", "Initial Screening Round") if webhook else "Initial Screening Round"
        
        # Get IST time (UTC + 5:30)
        from datetime import timedelta
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        
        # Store initial interview feedback record
        await interview_feedback_collection.insert_one({
            "feedback_id": session_id,
            "job_id": job_id,
            "applicant_email": email,
            "applicant_name": applicant.get("name", ""),
            "round": round_name,
            "type": "ai_interview",
            "session_id": session_id,
            "transcript": [],
            "recording_path": "",
            "started_at": ist_now,  # Store in IST
            "status": "incomplete"
        })
        
        # Get backend URL from environment
        backend_url = os.getenv("BACKEND_URL", "http://localhost:5555")
        ws_protocol = "wss" if backend_url.startswith("https") else "ws"
        ws_host = backend_url.replace("http://", "").replace("https://", "")
        
        print(f"✅ AI interview session created: {session_id} for {email} (job: {job_id})")
        
        return {
            "success": True,
            "session_id": session_id,
            "ws_url": f"{ws_protocol}://{ws_host}/ws/ai-interview/{session_id}",
            "jd_content": jd_text[:2000],  # Limit for initial load
            "resume_content": resume_text[:2000]  # Limit for initial load
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error starting AI interview: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


class SaveCalibrationRequest(BaseModel):
    """Request body for saving calibration images before AI interview."""
    session_id: str
    images: List[str]  # base64 data URLs (data:image/jpeg;base64,...)


@app.post("/api/ai-interview/save-calibration")
async def save_ai_interview_calibration(request: SaveCalibrationRequest):
    """
    Save calibration images (4 poses for face/eye tracking) before interview.
    Used to calibrate scores when reviewing the recording.
    """
    try:
        feedback = await interview_feedback_collection.find_one({
            "$or": [
                {"feedback_id": request.session_id},
                {"session_id": request.session_id}
            ]
        })
        if not feedback:
            print(f"❌ [SAVE-CALIBRATION] Session not found: session_id={request.session_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Interview session not found"
            )
        if len(request.images) != 4:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Exactly 4 calibration images are required"
            )

        job_id = feedback.get("job_id", "")
        email = feedback.get("applicant_email", "").replace("@", "_at_")
        calib_paths = []

        for i, data_url in enumerate(request.images):
            if "," in data_url:
                b64 = data_url.split(",", 1)[1]
            else:
                b64 = data_url
            try:
                content = base64.b64decode(b64)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 in image {i + 1}: {str(e)}"
                )
            try:
                path = upload_bytes_to_s3(
                    content,
                    key_prefix=f"interviews/{job_id}/{email}/calibration",
                    file_extension=f"_{i}.jpg",
                    content_type="image/jpeg",
                )
                calib_paths.append(path)
            except ValueError as e:
                print(f"❌ S3 not configured: {e}")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="File storage (S3) is not configured."
                )

        from datetime import timedelta
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        await interview_feedback_collection.update_one(
            {"feedback_id": request.session_id},
            {"$set": {"calibration_images": calib_paths, "updated_at": ist_now}}
        )
        print(f"✅ Calibration images saved for session {request.session_id}")
        return {"success": True, "count": len(calib_paths)}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving calibration: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.websocket("/ws/ai-interview/{session_id}")
async def ai_interview_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for AI interview real-time audio streaming.
    """
    await websocket.accept()
    print(f"🔗 AI Interview WebSocket connected: {session_id}")
    
    try:
        # Get interview feedback record
        feedback = await interview_feedback_collection.find_one({"feedback_id": session_id})
        
        if not feedback:
            await websocket.close(code=1008, reason="Session not found")
            return
        
        # Get JD and resume content for the agent
        job_id = feedback.get("job_id")
        email = feedback.get("applicant_email")
        
        # Fetch job and applicant details (similar to start endpoint)
        job = None
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job = await collection.find_one({"job_id": job_id})
            if job:
                break
        
        applicant = await job_applied_collection.find_one({
            "job_id": job_id,
            "email": email
        })
        
        # Extract JD text (supports PDF, DOCX, TXT)
        jd_text = _read_text_from_static_file_or_s3(static_dir, job.get("file_path", "")) if job else ""
        if not jd_text and job:
            jd_text = f"""Job Role: {job.get('role', 'Info not given')}
Location: {job.get('location', 'Info not given')}
Job Type: {job.get('job_type', 'Info not given')}
Notes: {job.get('notes', 'Info not given')}"""
        
        # Extract resume text (supports PDF, DOCX, TXT); fallback to user_data if job_applied has no resume_url
        resume_text = ""
        if applicant:
            resume_text = _read_text_from_static_file_or_s3(static_dir, applicant.get("resume_url", ""))
            if not resume_text and applicant.get("email"):
                user_profile = await user_data_collection.find_one({"user_email": applicant["email"]})
                if user_profile:
                    resume_text = _read_text_from_static_file_or_s3(
                        static_dir,
                        user_profile.get("resumeUrl") or user_profile.get("resume_url") or ""
                    )
        
        # Verify OpenAI API key is loaded
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ OPENAI_API_KEY not found in environment")
            await websocket.close(code=1011, reason="OpenAI API key not configured")
            return
        
        print(f"✅ OpenAI API Key loaded: {openai_key[:20]}...")
        
        # Create MongoDB storage
        storage = MongoDBTranscriptStorage(interview_feedback_collection)
        
        # Create AI interview agent
        agent = AIInterviewAgent(
            openai_api_key=openai_key,
            storage=storage,
            jd_content=jd_text,
            resume_content=resume_text
        )
        
        print(f"✅ AIInterviewAgent initialized for session: {session_id}")
        print(f"   JD length: {len(jd_text)} chars")
        print(f"   Resume length: {len(resume_text)} chars")
        if not jd_text:
            print("⚠️ JD text empty: model will use job fields only. Check job.file_path and static file.")
        if not resume_text:
            print("⚠️ Resume text empty: model will have no resume. Check applicant.resume_url / user_data.resumeUrl and static file.")
        
        # Start the interview
        print(f"🎤 Starting AI interview connection...")
        await agent.connect(websocket, session_id=session_id, timeout_seconds=1200)
        print(f"✅ AI interview completed for session: {session_id}")
        
    except WebSocketDisconnect:
        print(f"🔌 AI Interview WebSocket disconnected: {session_id}")
    except Exception as e:
        print(f"❌ Error in AI interview WebSocket: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass


@app.post("/api/ai-interview/save-recording")
async def save_ai_interview_recording(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Save AI interview recording (WebM file).
    """
    try:
        # Get interview feedback record
        feedback = await interview_feedback_collection.find_one({"feedback_id": session_id})
        
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Interview session not found"
            )
        
        job_id = feedback.get("job_id")
        email = feedback.get("applicant_email")
        email_safe = email.replace("@", "_at_")
        content = await file.read()
        filename = f"interview_{session_id}.webm"
        try:
            relative_path = upload_bytes_to_s3(
                content,
                key_prefix=f"interviews/{job_id}/{email_safe}",
                file_extension=".webm",
                content_type="video/webm",
            )
        except ValueError as e:
            print(f"❌ S3 not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="File storage (S3) is not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME and AWS_REGION in .env."
            )
        
        # Get IST time (UTC + 5:30)
        from datetime import timedelta
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        
        await interview_feedback_collection.update_one(
            {"feedback_id": session_id},
            {
                "$set": {
                    "recording_path": relative_path,
                    "updated_at": ist_now  # Store in IST
                }
            }
        )
        
        print(f"✅ AI interview recording saved: {relative_path}")
        
        return {
            "success": True,
            "recording_path": relative_path,
            "filename": filename
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving AI interview recording: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/ai-interview/save-camera-recording")
async def save_ai_interview_camera_recording(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    live_tracked: str = Form(default="false"),
):
    """
    Save AI interview camera recording (WebM) for face/eye tracking.
    """
    try:
        feedback = await interview_feedback_collection.find_one({"feedback_id": session_id})
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Interview session not found"
            )
        job_id = feedback.get("job_id")
        email = feedback.get("applicant_email")
        email_safe = email.replace("@", "_at_")
        content = await file.read()
        try:
            camera_path = upload_bytes_to_s3(
                content,
                key_prefix=f"interviews/{job_id}/{email_safe}/camera",
                file_extension=".webm",
                content_type="video/webm",
            )
        except ValueError as e:
            print(f"❌ S3 not configured: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="File storage (S3) is not configured."
            )
        from datetime import timedelta
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        update_fields = {"camera_recording_path": camera_path, "updated_at": ist_now}
        if live_tracked.lower() in ("true", "1", "yes"):
            update_fields["camera_live_tracked"] = True
        await interview_feedback_collection.update_one(
            {"feedback_id": session_id},
            {"$set": update_fields}
        )
        print(f"✅ AI interview camera recording saved: {camera_path}")
        return {"success": True, "camera_recording_path": camera_path}
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error saving camera recording: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def _run_face_eye_analysis(session_id: str, camera_url: str, calibration_urls: list, live_tracked: bool = False):
    """
    Background task: download camera video, run face/eye tracking, store metrics.
    If live_tracked=True, camera already has overlay from browser; only compute metrics, don't re-annotate.
    """
    print(f"[FACE_EYE] 🔄 Starting face/eye analysis for session {session_id} (live_tracked={live_tracked})")
    try:
        if not camera_url:
            print(f"[FACE_EYE] ⚠️ No camera_url provided, skipping")
            return
        print(f"[FACE_EYE] 📥 Downloading camera video from S3...")
        video_bytes = download_from_s3_url(camera_url)
        if not video_bytes:
            print(f"[FACE_EYE] ⚠️ Could not download camera video for {session_id}")
            return
        print(f"[FACE_EYE] 📥 Video downloaded ({len(video_bytes)} bytes), saving to temp file...")
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name
        print(f"[FACE_EYE] 📁 Temp video path: {tmp_path}")
        try:
            calib_count = len(calibration_urls) if calibration_urls else 0
            produce_annotated = not live_tracked
            print(f"[FACE_EYE] 🎬 Calling process_video (calibration: {calib_count}, produce_annotated={produce_annotated})...")
            metrics = await asyncio.to_thread(
                face_eye_process_video,
                tmp_path,
                calibration_image_urls=calibration_urls or None,
                download_fn=download_from_s3_url,
                produce_annotated_video=produce_annotated,
            )
            print(f"[FACE_EYE] 📊 process_video returned: error={metrics.get('error')}, score={metrics.get('score')}")
            if metrics.get("error"):
                print(f"[FACE_EYE] ⚠️ Face/eye analysis error for {session_id}: {metrics.get('error')}")
                return
            full_log = metrics.get("full_log")
            if full_log:
                try:
                    log_url = upload_bytes_to_s3(
                        full_log.encode("utf-8"),
                        key_prefix=f"interviews/{session_id}/face_eye",
                        file_extension=".txt",
                        content_type="text/plain",
                    )
                    metrics["full_log_file_url"] = log_url
                    print(f"[FACE_EYE] 📄 Full log saved to S3: {log_url}")
                except Exception as e:
                    print(f"[FACE_EYE] ⚠️ Failed to upload full log to S3: {e}")

            annotated_path = metrics.pop("annotated_video_path", None)
            annotated_url = None
            if live_tracked:
                pass
            elif annotated_path and os.path.exists(annotated_path):
                try:
                    annotated_url = upload_file_to_s3(
                        annotated_path,
                        key_prefix=f"interviews/{session_id}/face_eye",
                        file_extension=".mp4",
                        content_type="video/mp4",
                    )
                    metrics["annotated_video_url"] = annotated_url
                    print(f"[FACE_EYE] 🎬 Annotated video saved to S3: {annotated_url}")
                except Exception as e:
                    print(f"[FACE_EYE] ⚠️ Failed to upload annotated video to S3: {e}")
                finally:
                    try:
                        os.unlink(annotated_path)
                    except Exception:
                        pass

            camera_url = camera_url.strip() if camera_url else ""
            if not live_tracked and annotated_url and camera_url and is_s3_url(camera_url):
                deleted = delete_from_s3_url(camera_url)
                if deleted:
                    print(f"[FACE_EYE] 🗑️ Deleted raw camera video from S3")
                else:
                    print(f"[FACE_EYE] ⚠️ Could not delete raw camera video")

            update_doc = {"face_eye_metrics": metrics}
            if annotated_url:
                update_doc["camera_recording_path"] = annotated_url
            print(f"[FACE_EYE] 💾 Storing face_eye_metrics in MongoDB...")
            await interview_feedback_collection.update_one(
                {"feedback_id": session_id},
                {"$set": update_doc}
            )
            print(f"[FACE_EYE] ✅ Face/eye metrics stored for {session_id}: score={metrics.get('score')}")
        finally:
            try:
                os.unlink(tmp_path)
                print(f"[FACE_EYE] 🗑️ Temp file cleaned up")
            except Exception as ex:
                print(f"[FACE_EYE] ⚠️ Failed to delete temp file: {ex}")
    except Exception as e:
        print(f"[FACE_EYE] ❌ Face/eye analysis failed for {session_id}: {e}")
        import traceback
        traceback.print_exc()


@app.post("/api/ai-interview/complete")
async def complete_ai_interview(
    background_tasks: BackgroundTasks,
    session_id: str = Body(...),
    job_id: str = Body(...),
    email: str = Body(...),
):
    """
    Mark AI interview as complete.
    Move round to previous_rounds and update candidate status.
    """
    try:
        # Get IST time (UTC + 5:30)
        from datetime import timedelta
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        
        # Update interview feedback status
        await interview_feedback_collection.update_one(
            {"feedback_id": session_id},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": ist_now  # Store in IST
                }
            }
        )
        
        # Get feedback record to find round info
        feedback = await interview_feedback_collection.find_one({"feedback_id": session_id})
        
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Interview feedback not found"
            )
        
        round_name = feedback.get("round", "Initial Screening Round")
        recording_path = feedback.get("recording_path", "")
        applicant_name = feedback.get("applicant_name", "Unknown")
        
        # Get started and completed timestamps
        started_at = feedback.get("started_at", datetime.utcnow())
        completed_at = feedback.get("completed_at", datetime.utcnow())
        
        # Convert UTC to IST (UTC + 5:30)
        from datetime import timedelta
        if isinstance(started_at, datetime):
            ist_time = started_at + timedelta(hours=5, minutes=30)
        else:
            ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
        
        # Format date and time (12-hour format with AM/PM in IST - Hyderabad time)
        interview_date = ist_time.strftime("%Y-%m-%d")
        interview_time = ist_time.strftime("%I:%M %p")  # 12-hour format like "4:50 PM"
        
        # Create round entry for previous_rounds (without attended/outcome for AI interviews)
        round_entry = {
            "round": round_name,
            "type": "ai_interview",
            "feedback_id": session_id,
            "interview_date": interview_date,
            "interview_time": interview_time,
            "completed_at": completed_at,
            "status": "completed",
            "recording_path": recording_path  # Include video path
        }
        
        # Update job_applied: move to previous_rounds and update status
        await job_applied_collection.update_one(
            {"job_id": job_id, "email": email},
            {
                "$push": {"previous_rounds": round_entry},
                "$set": {
                    "status": "selected_for_interview",  # Ready for next round
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        # Update webhook status if exists
        await interview_webhook_collection.update_many(
            {
                "applicantEmail": email,
                "job_id": job_id,
                "is_ai_interview": True,
                "status": {"$ne": "completed"}
            },
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow()
                }
            }
        )
        
        # Run LLM evaluation (JD + transcript -> score + suggestion) and store in feedback
        transcript_list = feedback.get("transcript") or []
        jd_text = ""
        job_for_jd = None
        for coll in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job_for_jd = await coll.find_one({"job_id": job_id})
            if job_for_jd:
                break
        if job_for_jd:
            jd_text = _read_text_from_static_file_or_s3(static_dir, job_for_jd.get("file_path", ""))
            if not jd_text:
                jd_text = f"""Job Role: {job_for_jd.get('role', 'Info not given')}
Location: {job_for_jd.get('location', 'Info not given')}
Job Type: {job_for_jd.get('job_type', 'Info not given')}
Notes: {job_for_jd.get('notes', 'Info not given')}"""
        llm_eval = await _evaluate_ai_interview_with_llm(jd_text, transcript_list)
        if llm_eval:
            await interview_feedback_collection.update_one(
                {"feedback_id": session_id},
                {"$set": {"llm_evaluation": llm_eval}}
            )
            print(f"✅ AI interview LLM evaluation stored: score={llm_eval.get('score')} for {session_id}")

        # Run face/eye analysis in background (on camera recording)
        camera_recording_path = feedback.get("camera_recording_path", "")
        calibration_images = feedback.get("calibration_images", [])
        camera_live_tracked = feedback.get("camera_live_tracked", False)
        if camera_recording_path and background_tasks:
            print(f"[FACE_EYE] 📋 Scheduling background task for session {session_id} (calibration: {len(calibration_images)}, live_tracked: {camera_live_tracked})")
            background_tasks.add_task(
                _run_face_eye_analysis,
                session_id,
                camera_recording_path,
                calibration_images,
                camera_live_tracked,
            )
        
        print(f"✅ AI interview completed: {session_id} for {email} (job: {job_id})")
        
        return {
            "success": True,
            "message": "AI interview completed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error completing AI interview: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/interview-feedback/{feedback_id}")
async def get_interview_feedback(
    feedback_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get interview feedback by feedback_id (for viewing AI interview transcripts).
    """
    try:
        feedback = await interview_feedback_collection.find_one({"feedback_id": feedback_id})
        
        if not feedback:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Interview feedback not found"
            )
        
        # Convert ObjectId to string for JSON serialization
        if '_id' in feedback:
            feedback['_id'] = str(feedback['_id'])
        
        return feedback
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching interview feedback: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/storage/presigned-url")
async def get_presigned_storage_url(
    url: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Return a presigned URL for private S3 content (e.g. interview videos).
    Requires auth. Only allows URLs for our bucket.
    """
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="url query param required")
    url = url.strip()
    if not is_s3_url(url):
        raise HTTPException(status_code=400, detail="Invalid S3 URL")
    from services.s3_service import _parse_s3_object_url, _parse_s3_website_url
    parsed = _parse_s3_website_url(url) or _parse_s3_object_url(url)
    if not parsed:
        raise HTTPException(status_code=400, detail="Could not parse S3 URL")
    bucket, _ = parsed
    if bucket != get_bucket_name():
        raise HTTPException(status_code=403, detail="URL must be for configured bucket")
    presigned = generate_presigned_url(url, expires_in=3600)
    if not presigned:
        raise HTTPException(status_code=502, detail="Failed to generate presigned URL")
    return {"url": presigned}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5555)
