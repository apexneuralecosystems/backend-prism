import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Load environment variables
backend_dir = Path(__file__).parent.parent
env_path = backend_dir / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=True)
else:
    load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/meetings.space.created']

def create_meet_space():
    """
    Create a Google Meet space and return the meeting URI using the Meet REST API.
    Uses token from environment and refreshes it if expired.
    """
    creds = None

    # Load token from environment variable
    token_json_str = os.getenv("GOOGLE_TOKEN_JSON")
    if not token_json_str:
        raise ValueError(
            "GOOGLE_TOKEN_JSON not found in environment variables. "
            "Please provide a valid Google OAuth token."
        )

    try:
        token_data = json.loads(token_json_str)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid GOOGLE_TOKEN_JSON format: {e}")

    # Refresh token if expired
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                raise RuntimeError(
                    f"Token refresh failed: {e}. "
                    "Please update GOOGLE_TOKEN_JSON with a new token."
                )
        else:
            raise ValueError(
                "Token is invalid and cannot be refreshed. "
                "Please update GOOGLE_TOKEN_JSON with a valid token."
            )

    # Build Meet service via REST API and create a space
    service = build("meet", "v2", credentials=creds, cache_discovery=False)
    response = service.spaces().create(body={}).execute()
    meeting_uri = response.get("meetingUri")

    if not meeting_uri:
        raise RuntimeError("Failed to create meeting URI from Google Meet API response.")

    return meeting_uri
