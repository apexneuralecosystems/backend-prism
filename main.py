from fastapi import FastAPI, HTTPException, Body, Depends, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta
import random
import motor.motor_asyncio
import uuid
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
open_jobs_collection = db["open-jobs"]
ongoing_jobs_collection = db["ongoing-jobs"]
closed_jobs_collection = db["closed-jobs"]
job_applied_collection = db["job-applied"]
refresh_tokens_collection = db["refresh_tokens"]
interview_webhook_collection = db["interview_webhooks"]
interview_feedback_collection = db["interview-feedback"]
offer_webhook_collection = db["offer_webhooks"]

# FastAPI app
app = FastAPI(
    title="PRISM API - Complete Backend",
    version="2.0.0",
    description="Unified backend with Authentication, User Profiles, and Demo Requests"
)

# CORS - Configurable via environment variable
CORS_ORIGINS_STR = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
CORS_ORIGINS = [origin.strip() for origin in CORS_ORIGINS_STR.split(",") if origin.strip()]
print(f"🔍 CORS allowed origins: {CORS_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
static_dir = backend_dir / "static"
static_dir.mkdir(exist_ok=True)
(static_dir / "uploads").mkdir(exist_ok=True)
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


# ==================== HELPER FUNCTIONS ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and get current user"""
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
    
    return user


# ==================== AUTHENTICATION ENDPOINTS ====================

@app.post("/api/auth/signup", status_code=status.HTTP_200_OK)
async def signup(user_data: UserSignup):
    """
    User/Organization Signup - Send OTP to email
    
    IMPORTANT: This endpoint ONLY sends OTP. Account creation happens ONLY
    after successful OTP verification in /api/auth/verify-otp endpoint.
    """
    try:
        # Check if user already exists
        existing_user = await users_collection.find_one({
            "email": user_data.email,
            "user_type": user_data.user_type
        })
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{user_data.user_type.capitalize()} already exists with this email"
            )
        
        # Generate and store OTP (NO account creation here)
        otp = generate_otp()
        await store_otp(db, user_data.email, otp)
        
        # Send OTP email
        from services.email_service import send_otp_email
        await send_otp_email(user_data.email, otp, "signup")
        
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
        
        # Check if user already exists (in case of race condition or duplicate signup)
        existing_user = await users_collection.find_one({
            "email": email,
            "user_type": user_type
        })
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{user_type.capitalize()} account already exists with this email"
            )
        
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
        
        # Generate tokens
        token_data = {"sub": login_data.email, "user_type": login_data.user_type}
        access_token = create_access_token(token_data)
        refresh_token = create_refresh_token(token_data)
        
        # Store refresh token
        await refresh_tokens_collection.insert_one({
            "user_id": str(user["_id"]),
            "token": refresh_token,
            "created_at": datetime.utcnow()
        })
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user={
                "id": str(user["_id"]),
                "email": user["email"],
                "name": user["name"],
                "user_type": user["user_type"]
            }
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
        
        # Generate new access token
        token_data = {
            "sub": payload.get("sub"),
            "user_type": payload.get("user_type")
        }
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
    Upload a file and return the URL
    """
    try:
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        
        # Save file to uploads directory
        file_path = static_dir / "uploads" / unique_filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Return URL
        file_url = f"/static/uploads/{unique_filename}"
        
        return {
            "success": True,
            "url": file_url,
            "filename": file.filename
        }
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
        
        # Remove _id if it exists
        if "_id" in profile_data:
            profile_data.pop("_id")

        # Update organization data by email
        profile_data["email"] = current_user["email"]
        profile_data["updated_at"] = datetime.utcnow()
        
        # Ensure employees_details is an array if provided
        if "employees_details" in profile_data and not isinstance(profile_data["employees_details"], list):
            profile_data["employees_details"] = []
        
        # Update existing document or create if not exists
        result = await organization_data_collection.update_one(
            {"email": current_user["email"]},
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
    """
    try:
        # Verify user is an organization
        if current_user.get("user_type") != "organization":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only organizations can access this endpoint"
            )
        
        profile = await organization_data_collection.find_one({
            "email": current_user["email"]
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
    job_package_lpa: float = Form(...),
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

        # Get organization data for company info
        org_data = await organization_data_collection.find_one({
            "email": current_user["email"]
        })

        if not org_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Organization profile not found. Please complete your profile first."
            )

        # Handle file upload
        file_path = ""
        if jd_file and jd_file.filename:
            # Generate unique filename
            file_ext = os.path.splitext(jd_file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"

            # Save file to jds directory
            jds_dir = static_dir / "files" / "jds"
            jds_dir.mkdir(parents=True, exist_ok=True)
            file_full_path = jds_dir / unique_filename

            with open(file_full_path, "wb") as buffer:
                shutil.copyfileobj(jd_file.file, buffer)

            file_path = f"/static/files/jds/{unique_filename}"

        # Generate job ID
        job_id = f"job_{uuid.uuid4()}"

        # Create job document
        job_doc = {
            "job_id": job_id,
            "company": {
                "name": org_data.get("name", current_user.get("name", "")),
                "email": current_user["email"]
            },
            "role": role,
            "file_path": file_path,
            "location": location,
            "number_of_openings": number_of_openings,
            "application_close_date": application_close_date,
            "job_package_lpa": job_package_lpa,
            "job_type": job_type,
            "notes": notes,
            "applied_candidates": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        # Save to open-jobs collection
        result = await open_jobs_collection.insert_one(job_doc)

        print(f"✅ [POST /api/organization-jobpost] Job created: {job_id}")

        return {
            "success": True,
            "message": "Job posting created successfully",
            "job_id": job_id
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

        # Get open jobs for this organization
        jobs_cursor = open_jobs_collection.find({
            "company.email": current_user["email"]
        }).sort("created_at", -1)

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

        # Find the job in ongoing-jobs collection
        job = await ongoing_jobs_collection.find_one({
            "job_id": job_id,
            "company.email": current_user["email"]
        })

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found or not in ongoing status"
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
        valid_statuses = ["applied", "decision_pending", "selected_for_interview", "rejected", "selected", "processing", "invitation_sent"]
        if not new_status or new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
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
                    "company.email": current_user["email"],
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

        # Get ongoing jobs for this organization
        jobs_cursor = ongoing_jobs_collection.find({
            "company.email": current_user["email"]
        }).sort("created_at", -1)

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

        # Get closed jobs for this organization
        jobs_cursor = closed_jobs_collection.find({
            "company.email": current_user["email"]
        }).sort("closed_at", -1)

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

        # Check if user has already applied
        existing_job = await open_jobs_collection.find_one({
            "job_id": job_id,
            "applied_candidates.email": current_user["email"]
        })

        if existing_job:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You have already applied for this job"
            )

        # Get job details to find organization and job description
        job_details = await open_jobs_collection.find_one({"job_id": job_id})
        if not job_details:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        org_email = job_details.get("company", {}).get("email", "")
        job_description = ""  # Will be extracted from JD file if available
        
        # Get user profile to fetch resume path and parsed resume data
        user_profile = await user_data_collection.find_one({"user_email": current_user["email"]})
        
        # Get resume_url from request body first, fallback to profile
        resume_url = application_data.get("resume_url", "")
        if not resume_url and user_profile:
            resume_url = user_profile.get("resumeUrl", "") or user_profile.get("resume_url", "")
        
        print(f"📄 [APPLY] Resume URL for {current_user['email']}: {resume_url}")
        
        if not resume_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resume not found. Please upload your resume in your profile before applying."
            )
        
        # Always parse resume freshly on each application for accurate matching
        # This ensures LLM is called every time user applies
        parsed_resume_data = None
        try:
            print(f"🤖 [APPLY] Starting LLM resume parsing for {current_user['email']}...")
            
            # Read resume file from path
            resume_path = static_dir / resume_url.lstrip("/static/")
            if not resume_path.exists():
                raise Exception(f"Resume file not found at path: {resume_path}")
            
            with open(resume_path, "rb") as f:
                resume_content = f.read()
            
            parser = ResumeParser()
            resume_text = parser.extract_text_from_pdf(resume_content)
            parsed_resume_data = parser.parse(resume_text)
            
            print(f"✅ [APPLY] LLM resume parsing completed successfully for {current_user['email']}!")
            
            # Save parsed data back to user profile
            await user_data_collection.update_one(
                {"user_email": current_user["email"]},
                {"$set": {"parsed_resume_data": parsed_resume_data}}
            )
        except Exception as e:
            print(f"❌ [APPLY] Error parsing resume with LLM: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process resume with AI. Please ensure your resume is uploaded correctly. Error: {str(e)}"
            )
        
        # Prepare candidate data with resume URL
        candidate_data = {
            "name": application_data.get("name", current_user.get("name", "")),
            "email": current_user["email"],
            "resume_url": resume_url
        }
        
        # Run comparator agent if we have parsed resume data and organization employees
        additional_details = ""
        if parsed_resume_data and org_email:
            try:
                print(f"🔍 [APPLY] Running comparator agent for {current_user['email']}...")
                
                # Get organization employees data
                org_data = await organization_data_collection.find_one({"email": org_email})
                employees_data = []
                
                if org_data and org_data.get("employees_details"):
                    for emp in org_data["employees_details"]:
                        if emp.get("parsed_resume_data"):
                            employees_data.append(emp["parsed_resume_data"])
                    print(f"📊 [APPLY] Found {len(employees_data)} employees for comparison")
                
                # Get job description from JD file if available
                jd_file_path = job_details.get("file_path", "")
                job_description = ""
                if jd_file_path:
                    try:
                        jd_path = static_dir / jd_file_path.lstrip("/static/")
                        if jd_path.exists():
                            with open(jd_path, "rb") as f:
                                jd_content = f.read()
                            # Try to extract text from PDF
                            try:
                                pdf_reader = PyPDF2.PdfReader(io.BytesIO(jd_content))
                                job_description = ""
                                for page in pdf_reader.pages:
                                    job_description += page.extract_text() + "\n"
                                print(f"📄 [APPLY] Extracted JD from PDF")
                            except:
                                # If not PDF, try reading as text
                                job_description = jd_content.decode('utf-8', errors='ignore')
                                print(f"📄 [APPLY] Read JD as text file")
                    except Exception as e:
                        print(f"⚠️ [APPLY] Warning: Could not read JD file: {e}")
                
                # Run comparator agent if we have employees data
                if employees_data:
                    print(f"🚀 [APPLY] Starting comparator agent with {len(employees_data)} employees...")
                    comparator = ComparatorAgent(employees_data)
                    if job_description:
                        print(f"📝 [APPLY] Running full comparison with JD and employees...")
                        additional_details = comparator.process(
                            candidate_data=parsed_resume_data,
                            job_description=job_description,
                            additional_info={}
                        )
                        print(f"✅ [APPLY] Comparator agent completed successfully!")
                    else:
                        print(f"📝 [APPLY] Running employee comparison only (no JD available)...")
                        # If no JD, just compare with employees
                        employee_comparison = comparator._compare_with_employees(parsed_resume_data)
                        # Format just employee comparison
                        from services.output_formatter import OutputFormatter
                        formatter = OutputFormatter()
                        additional_details = formatter.format(
                            parsed_resume_data,
                            employee_comparison,
                            {"jd_relevance_score": 0, "ai_suggestion": "Job description not available"},
                            {}
                        )
                        print(f"✅ [APPLY] Employee comparison completed successfully!")
                else:
                    print(f"⚠️ [APPLY] No employees data available for comparison")
            except Exception as e:
                print(f"❌ [APPLY] Error running comparator agent: {e}")
                import traceback
                traceback.print_exc()
        
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
            "additional_details": additional_details,
            "applied_at": datetime.utcnow()
        }

        await job_applied_collection.insert_one(applied_job_data)

        print(f"✅ [PUT /api/organization-jobpost/{job_id}/apply] Application submitted by: {current_user['email']}")

        return {
            "success": True,
            "message": "Successfully applied for the job"
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
                    "location": "Not specified",
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

        # Add/update organization email and timestamp
        team_data["organization_email"] = current_user["email"]
        team_data["updated_at"] = datetime.utcnow()

        # Create or update the teams document for this organization in separate collection
        result = await organization_teams_collection.update_one(
            {"organization_email": current_user["email"]},
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

        # Get organization teams data from separate collection
        teams_data = await organization_teams_collection.find_one({
            "organization_email": current_user["email"]
        })

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
        
        # Get frontend URL from environment
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        
        # Check candidate's current status
        applicant_doc = await job_applied_collection.find_one({
            "job_id": request.job_id,
            "email": request.applicantEmail
        })
        
        candidate_status = applicant_doc.get("status") if applicant_doc else None
        
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
        
        # Create form link with webhook ID
        form_link = f"{frontend_url}/schedule-interview?webhook_id={webhook_id}&applicantEmail={request.applicantEmail}&applicantName={request.applicantName}&orgEmail={request.orgEmail}&orgName={request.orgName}&round={request.round}&team={request.team}"
        
        # Store webhook mapping first to ensure correct matching
        await interview_webhook_collection.insert_one({
            "webhook_id": webhook_id,
            "applicantEmail": request.applicantEmail,
            "applicantName": request.applicantName,
            "orgEmail": request.orgEmail,
            "orgName": request.orgName,
            "round": request.round,
            "team": request.team,
            "job_id": request.job_id,
            "location_type": request.location_type,
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
        
        # Get job details and JD file
        job_details = None
        jd_file_path = None
        
        # Check all job collections
        for collection in [open_jobs_collection, ongoing_jobs_collection, closed_jobs_collection]:
            job_details = await collection.find_one({"job_id": job_id})
            if job_details:
                jd_file_path_str = job_details.get("file_path", "")
                if jd_file_path_str:
                    jd_file_path = static_dir / jd_file_path_str.lstrip("/static/")
                    if not jd_file_path.exists():
                        jd_file_path = None
                break
        
        # Get applicant resume
        applicant_email = webhook_record.get("applicantEmail")
        user_profile = await user_data_collection.find_one({"user_email": applicant_email})
        resume_file_path = None
        
        if user_profile:
            resume_url = user_profile.get("resumeUrl") or user_profile.get("resume_url", "")
            if resume_url:
                resume_file_path = static_dir / resume_url.lstrip("/static/")
                if not resume_file_path.exists():
                    resume_file_path = None
        
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
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
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
        
        # Send confirmation emails (combined - includes feedback form link for interviewer)
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
            resume_file_path=str(resume_file_path) if resume_file_path else None
        )
        
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
        
        # Save offer letter file
        file_ext = os.path.splitext(offer_letter.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        offers_dir = static_dir / "files" / "offers"
        offers_dir.mkdir(parents=True, exist_ok=True)
        file_full_path = offers_dir / unique_filename
        
        with open(file_full_path, "wb") as buffer:
            shutil.copyfileobj(offer_letter.file, buffer)
        
        file_path = f"/static/files/offers/{unique_filename}"
        
        # Create webhook ID for offer response
        webhook_id = str(uuid.uuid4())
        frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
        form_link = f"{frontend_url}/offer-response?offer_id={webhook_id}"
        
        # Store offer webhook
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
        
        # Get job role from ongoing-jobs collection
        job_details = await ongoing_jobs_collection.find_one({"job_id": job_id})
        job_role = job_details.get("role", "the position") if job_details else "the position"
        
        # Send email with offer letter
        from services.email_service import send_offer_letter_email
        success = await send_offer_letter_email(
            applicant_email=applicantEmail,
            applicant_name=applicantName,
            org_name=orgName,
            job_role=job_role,
            offer_letter_path=str(file_full_path),
            form_link=form_link
        )
        
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
