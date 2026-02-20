"""
S3 file storage service. All uploads go to AWS S3; only S3 Object URLs (HTTPS) are stored in MongoDB.
Uses env: S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION.
Bucket: prism-bucket-10, Region: ap-south-2 (Hyderabad).
"""
import os
import uuid
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# Explicit region - must match bucket (prism-bucket-10 in ap-south-2)
S3_REGION = "ap-south-2"
S3_BUCKET = "prism-bucket-10"


def _get_credentials():
    """Resolve AWS credentials from env."""
    access_key = os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("YOUR_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("YOUR_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION") or os.getenv("S3_REGION") or S3_REGION
    return access_key, secret_key, region


def get_bucket_name() -> str:
    """Return S3 bucket name (prism-bucket-10)."""
    return os.getenv("S3_BUCKET_NAME", S3_BUCKET).strip()


def _get_s3_client():
    """Create boto3 S3 client with explicit ap-south-2 regional endpoint (required for presigned URLs)."""
    access_key, secret_key, _ = _get_credentials()
    if not access_key or not secret_key:
        raise ValueError("AWS credentials not set: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION") or os.getenv("S3_REGION") or S3_REGION
    region = str(region).strip().strip('"').strip("'")
    return boto3.client(
        "s3",
        aws_access_key_id=access_key.strip(),
        aws_secret_access_key=secret_key.strip(),
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def _object_url_for_key(bucket: str, key: str) -> str:
    """Build S3 Object URL (HTTPS only). Format: https://bucket.s3.region.amazonaws.com/key"""
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def is_s3_url(path_or_url: str) -> bool:
    """Return True if path_or_url is an S3 website or S3 object URL."""
    if not path_or_url or not isinstance(path_or_url, str):
        return False
    p = path_or_url.strip()
    if p.startswith("http://") or p.startswith("https://"):
        return "s3" in p and ("amazonaws.com" in p or "s3-website" in p)
    return False


def upload_bytes_to_s3(
    content: bytes,
    key_prefix: str,
    file_extension: str,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload bytes to S3 with a unique key. Returns the S3 object URL (HTTPS).
    key_prefix e.g. "uploads", "jds", "offers", "interviews/job_xxx/email_at_domain"
    """
    bucket = get_bucket_name()
    access_key, secret_key, _ = _get_credentials()
    if not access_key or not secret_key:
        raise ValueError("AWS credentials not set")

    unique_id = uuid.uuid4().hex
    key = f"{key_prefix.rstrip('/')}/{unique_id}{file_extension}"

    s3 = _get_s3_client()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    # Inline disposition so PDFs/images open in browser tab instead of downloading
    is_displayable = (
        (content_type and any(content_type.startswith(t) for t in ("application/pdf", "image/")))
        or (file_extension and file_extension.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp"))
    )
    if is_displayable:
        extra["ContentDisposition"] = "inline"
    s3.put_object(Bucket=bucket, Key=key, Body=content, **extra)
    return _object_url_for_key(bucket, key)


def upload_file_to_s3(
    local_path: str,
    key_prefix: str,
    file_extension: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload a local file to S3. Returns the S3 object URL (HTTPS).
    If file_extension not provided, uses the file's suffix.
    """
    from pathlib import Path
    path = Path(local_path)
    if not path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")
    ext = file_extension or path.suffix or ""
    content = path.read_bytes()
    return upload_bytes_to_s3(content, key_prefix, ext, content_type=content_type)


def _parse_s3_website_url(url: str) -> Optional[tuple]:
    """
    Parse S3 website URL (legacy HTTP or HTTPS) to (bucket, key).
    e.g. http(s)://prism-bucket-10.s3-website.ap-south-2.amazonaws.com/uploads/abc.pdf -> ("prism-bucket-10", "uploads/abc.pdf")
    """
    if not url or "s3-website" not in url or "amazonaws.com" not in url:
        return None
    try:
        # http://bucket.s3-website.region.amazonaws.com/key
        after_proto = url.split("://", 1)[-1]
        host = after_proto.split("/", 1)[0]
        path = after_proto.split("/", 1)[1] if "/" in after_proto else ""
        if ".s3-website." in host:
            bucket = host.split(".s3-website.")[0]
            return (bucket, path)
        return None
    except Exception:
        return None


def _parse_s3_object_url(url: str) -> Optional[tuple]:
    """
    Parse S3 object URL to (bucket, key).
    e.g. https://bucket.s3.region.amazonaws.com/key or https://s3.region.amazonaws.com/bucket/key
    """
    if not url or "amazonaws.com" not in url:
        return None
    try:
        after_proto = url.split("://", 1)[-1]
        parts = after_proto.split("/", 1)
        host = parts[0]
        path = parts[1] if len(parts) > 1 else ""
        if ".s3." in host and ".amazonaws.com" in host:
            bucket = host.split(".s3.")[0]
            return (bucket, path)
        if host.startswith("s3.") and ".amazonaws.com" in host and "/" in path:
            bucket, key = path.split("/", 1)
            return (bucket, key)
        return None
    except Exception:
        return None


def to_object_url(url: str) -> str:
    """
    Convert any S3 URL to Object URL format (HTTPS), using ap-south-2.
    Legacy s3-website -> https://bucket.s3.ap-south-2.amazonaws.com/key
    """
    if not url or not isinstance(url, str):
        return url or ""
    u = url.strip()
    if not u.startswith("http") or "amazonaws.com" not in u:
        return u
    parsed = _parse_s3_website_url(u) or _parse_s3_object_url(u)
    if parsed:
        bucket, key = parsed
        return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"
    return u.replace("http://", "https://") if u.startswith("http://") else u


def delete_from_s3_url(url: str) -> bool:
    """
    Delete object from S3 by URL. Returns True on success, False on failure.
    """
    if not url or not is_s3_url(url):
        return False
    parsed = _parse_s3_website_url(url) or _parse_s3_object_url(url)
    if not parsed:
        return False
    bucket, key = parsed
    try:
        s3 = _get_s3_client()
        s3.delete_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        print(f"⚠️ S3 delete failed for {url}: {e}")
        return False
    except Exception as e:
        print(f"⚠️ S3 delete error: {e}")
        return False


def generate_presigned_url(url: str, expires_in: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for an S3 object. Returns a temporary URL that allows
    unauthenticated read access. Use for private bucket content (e.g. video playback).
    """
    if not url or not is_s3_url(url):
        return None
    parsed = _parse_s3_website_url(url) or _parse_s3_object_url(url)
    if not parsed:
        return None
    bucket, key = parsed
    try:
        s3 = _get_s3_client()
        presigned = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_in,
        )
        return presigned
    except ClientError as e:
        print(f"⚠️ S3 presign failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ S3 presign error: {e}")
        return None


def download_from_s3_url(url: str) -> Optional[bytes]:
    """
    Download file content from an S3 URL (website or object URL). Returns bytes or None on failure.
    """
    if not url or not is_s3_url(url):
        parsed = None
    else:
        parsed = _parse_s3_website_url(url) or _parse_s3_object_url(url)
    if not parsed:
        return None
    bucket, key = parsed
    try:
        s3 = _get_s3_client()
        resp = s3.get_object(Bucket=bucket, Key=key)
        return resp["Body"].read()
    except ClientError as e:
        print(f"⚠️ S3 download failed for {url}: {e}")
        return None
    except Exception as e:
        print(f"⚠️ S3 download error: {e}")
        return None
