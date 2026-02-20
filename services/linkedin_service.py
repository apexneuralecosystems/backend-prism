"""
LinkedIn OAuth credential storage and posting (org-only).
Credentials are stored by profile_id (org email). Posting uses owner's token.
"""
import os
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta
from urllib.parse import quote
import time
import requests

# Directory for social credentials (per org)
_backend_dir = Path(__file__).resolve().parent.parent
_credentials_dir = _backend_dir / "data" / "social_credentials"


def _profile_id_to_filename(profile_id: str, platform: str = "linkedin") -> str:
    """Sanitize profile_id (e.g. org email) for use as filename."""
    safe = re.sub(r"[^\w\-.]", "_", profile_id)
    safe = safe.replace(".", "_").replace("@", "_at_")
    return f"{safe}_{platform}.json"


def _ensure_credentials_dir() -> Path:
    _credentials_dir.mkdir(parents=True, exist_ok=True)
    return _credentials_dir


def save_social_credential(profile_id: str, platform: str, data: Dict[str, Any]) -> None:
    """Save LinkedIn (or other) credentials for a profile (org email)."""
    if not profile_id or platform != "linkedin":
        return
    _ensure_credentials_dir()
    path = _credentials_dir / _profile_id_to_filename(profile_id, platform)
    expires_in = data.get("expires_in", 5184000)  # 60 days default
    expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat() + "Z"
    payload = {
        "access_token": data.get("access_token"),
        "refresh_token": data.get("refresh_token"),
        "expires_in": expires_in,
        "token_expires_at": expires_at,
        "platform_user_id": data.get("platform_user_id"),
        "platform_username": data.get("platform_username"),
        "platform_email": data.get("platform_email"),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def get_social_credential(profile_id: str, platform: str) -> Optional[Dict[str, Any]]:
    """Load credential for profile_id. Returns None if missing or invalid."""
    if not profile_id or platform != "linkedin":
        return None
    path = _credentials_dir / _profile_id_to_filename(profile_id, platform)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def is_token_expired(credential: Dict[str, Any]) -> bool:
    """Check if stored token is expired."""
    expires_at_str = credential.get("token_expires_at")
    if not expires_at_str:
        return False
    try:
        # Parse as naive UTC for comparison with datetime.utcnow()
        s = expires_at_str.replace("Z", "").strip()
        expires_at = datetime.fromisoformat(s)
        if expires_at.tzinfo:
            expires_at = expires_at.replace(tzinfo=None)
        return datetime.utcnow() >= expires_at
    except Exception:
        return True


def get_active_access_token(profile_id: str, platform: str) -> Optional[str]:
    """Return valid access token for profile_id, or None if missing/expired."""
    cred = get_social_credential(profile_id, platform)
    if not cred:
        return None
    if is_token_expired(cred):
        return None
    return cred.get("access_token")


def delete_social_credential(profile_id: str, platform: str) -> None:
    """Remove stored credential for profile_id."""
    if not profile_id or platform != "linkedin":
        return
    path = _credentials_dir / _profile_id_to_filename(profile_id, platform)
    if path.exists():
        path.unlink()


def post_to_linkedin(
    access_token: str,
    text: str,
    image_url: Optional[str] = None,
    article_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Post text to LinkedIn UGC API (v2).
    If article_url is provided, use shareMediaCategory ARTICLE so the URL appears as a link card.
    """
    if not access_token or not text:
        return {"success": False, "error": "Missing access_token or text"}

    user_id = _get_linkedin_user_id(access_token)
    if not user_id:
        return {"success": False, "error": "LinkedIn user id not found"}

    share_media_category = "NONE"
    media_block = []
    if article_url:
        share_media_category = "ARTICLE"
        media_block = [
            {
                "status": "READY",
                "originalUrl": article_url[:8192],
            }
        ]
    elif image_url:
        share_media_category = "IMAGE"
        # media_block stays [] (image upload not implemented)

    post_data = {
        "author": f"urn:li:person:{user_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": text},
                "shareMediaCategory": share_media_category,
                "media": media_block,
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }

    ugc_url = "https://api.linkedin.com/v2/ugcPosts"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
    }
    try:
        resp = requests.post(ugc_url, headers=headers, json=post_data, timeout=15)
        if resp.status_code == 201:
            body = resp.json()
            return {
                "success": True,
                "post_id": body.get("id"),
                "platform": "linkedin",
            }
        return {
            "success": False,
            "error": resp.text or f"HTTP {resp.status_code}",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# LinkedIn REST API version for Documents and Posts (use current active version)
LINKEDIN_REST_VERSION = "202510"


def _get_content_type_for_file(path: Path) -> str:
    """Return Content-Type for LinkedIn document upload."""
    suffix = (path.suffix or "").lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix in (".doc", ".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if suffix == ".docx" else "application/msword"
    return "application/octet-stream"


def _get_linkedin_user_id(access_token: str) -> Optional[str]:
    """Get LinkedIn person URN id (sub) from userinfo."""
    try:
        r = requests.get(
            "https://api.linkedin.com/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        return r.json().get("sub")
    except Exception:
        return None


def post_to_linkedin_with_document(
    access_token: str,
    local_file_path: Union[str, Path],
    commentary: str,
    document_title: str,
) -> Dict[str, Any]:
    """
    Upload a document (PDF/DOC/DOCX) to LinkedIn and create a post with it attached.
    Uses LinkedIn Documents API (initializeUpload -> upload -> create post).
    """
    if not access_token or not commentary:
        return {"success": False, "error": "Missing access_token or commentary"}
    path = Path(local_file_path) if not isinstance(local_file_path, Path) else local_file_path
    if not path.exists():
        return {"success": False, "error": "JD file not found"}
    if not document_title.strip():
        document_title = "Job Description.pdf"

    user_id = _get_linkedin_user_id(access_token)
    if not user_id:
        return {"success": False, "error": "LinkedIn user id not found"}

    owner_urn = f"urn:li:person:{user_id}"
    author_urn = owner_urn
    rest_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": LINKEDIN_REST_VERSION,
    }

    # 1. Initialize document upload
    try:
        init_resp = requests.post(
            "https://api.linkedin.com/rest/documents?action=initializeUpload",
            headers=rest_headers,
            json={"initializeUploadRequest": {"owner": owner_urn}},
            timeout=15,
        )
        if init_resp.status_code not in (200, 201):
            return {
                "success": False,
                "error": f"Document init failed: {init_resp.status_code} - {init_resp.text}",
            }
        init_data = init_resp.json()
        value = init_data.get("value") or init_data
        upload_url = value.get("uploadUrl")
        document_urn = value.get("document")
        if not upload_url or not document_urn:
            return {"success": False, "error": "No uploadUrl or document URN in init response"}
    except Exception as e:
        return {"success": False, "error": f"Document init: {e}"}

    # 2. Upload file to LinkedIn (with correct Content-Type)
    try:
        with open(path, "rb") as f:
            file_bytes = f.read()
        content_type = _get_content_type_for_file(path)
        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": content_type,
        }
        up_resp = requests.put(upload_url, data=file_bytes, headers=upload_headers, timeout=60)
        if up_resp.status_code not in (200, 201, 204):
            return {
                "success": False,
                "error": f"Document upload failed: {up_resp.status_code} - {up_resp.text[:500]}",
            }
    except Exception as e:
        return {"success": False, "error": f"Document upload: {e}"}

    # 2b. Wait for document to be AVAILABLE (LinkedIn processes async)
    document_urn_encoded = quote(document_urn, safe="")
    for _ in range(10):
        time.sleep(2)
        try:
            doc_resp = requests.get(
                f"https://api.linkedin.com/rest/documents/{document_urn_encoded}",
                headers=rest_headers,
                timeout=10,
            )
            if doc_resp.status_code == 200:
                doc_data = doc_resp.json()
                status_val = doc_data.get("status", "")
                if status_val == "AVAILABLE":
                    break
                if status_val == "PROCESSING_FAILED":
                    return {"success": False, "error": "Document processing failed on LinkedIn"}
        except Exception:
            pass

    # 3. Create post with document (REST Posts API)
    try:
        post_payload = {
            "author": author_urn,
            "commentary": commentary,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [],
                "thirdPartyDistributionChannels": [],
            },
            "content": {
                "media": {
                    "title": document_title[:255],
                    "id": document_urn,
                },
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        post_resp = requests.post(
            "https://api.linkedin.com/rest/posts",
            headers=rest_headers,
            json=post_payload,
            timeout=15,
        )
        if post_resp.status_code in (200, 201):
            post_id = post_resp.headers.get("x-restli-id") or (post_resp.json().get("id") if post_resp.text else None)
            return {"success": True, "post_id": post_id, "platform": "linkedin"}
        return {
            "success": False,
            "error": f"Post create failed: {post_resp.status_code} - {post_resp.text[:500]}",
        }
    except Exception as e:
        return {"success": False, "error": f"Post create: {e}"}
