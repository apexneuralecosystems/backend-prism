import os
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.apps import meet_v2

SCOPES = ['https://www.googleapis.com/auth/meetings.space.created']

def create_meet_space():
    """
    Create a Google Meet space and return the meeting URI
    """
    creds = None
    backend_dir = Path(__file__).parent.parent
    token_path = backend_dir / "token.json"
    credentials_path = backend_dir / "credentials.json"
    
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not credentials_path.exists():
                raise FileNotFoundError("credentials.json not found. Please set up Google OAuth credentials.")
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    client = meet_v2.SpacesServiceClient(credentials=creds)
    request = meet_v2.CreateSpaceRequest()
    response = client.create_space(request=request)
    
    return response.meeting_uri

