"""
AI Interview Service
Real-time bidirectional voice conversation using OpenAI Realtime API.
Handles AI-powered interviews with MongoDB storage for transcripts and recordings.
"""

import os
import json
import base64
import asyncio
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK
import logging
import time
import uuid
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import inspect

# Import RAG module from same directory
try:
    from rag import get_system_message
    print(f"✅ Successfully imported get_system_message from rag")
except ImportError as e:
    print(f"⚠️ Could not import from rag: {e}")
    # Fallback if rag module not available
    def get_system_message(phone_number: str = "", introduction: str = "", jd_content: str = "", resume_content: str = ""):
            return f"""You are an experienced HR interviewer conducting a job interview.
            
Based on the resume and job description provided, ask 5-6 targeted questions to evaluate the candidate.

=== CANDIDATE RESUME ===
{resume_content if resume_content else "No resume provided"}
=== END OF RESUME ===

=== JOB DESCRIPTION ===
{jd_content if jd_content else "No job description provided"}
=== END OF JOB DESCRIPTION ===

INTERVIEW REQUIREMENTS:
1) Introduce yourself and mention the specific role
2) Ask 5-6 relevant questions based on the JD and resume
3) Keep the tone professional and conversational
4) After 5-6 questions, say: "The interview is now complete. Thank you for your time. You can now end the interview."
5) ALL conversation MUST be in English ONLY
6) Ask only ONE question at a time. After the user answers, ask the next relevant question based only on that response. Never ask multiple questions together.
7) Always allow the candidate to fully complete their answer before speaking. Do not interrupt, overlap, or cut off the candidate at any time. Wait for a clear pause or confirmation that the candidate has finished speaking before asking the next question. After the candidate finishes, respond professionally and proceed with the next relevant question. Maintain a respectful, patient, and neutral tone throughout the interaction.

START NOW: Begin with your introduction and the first question."""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _websockets_connect_with_headers(url: str, headers: Dict[str, str], **kwargs):
    """
    Compatibility wrapper for `websockets.connect` across versions.

    `websockets` has changed the kwarg name for passing HTTP headers:
    - older versions: `extra_headers=...`
    - newer versions: `additional_headers=...`
    """
    connect_sig = inspect.signature(websockets.connect)
    params = connect_sig.parameters

    if "additional_headers" in params:
        return websockets.connect(url, additional_headers=headers, **kwargs)
    if "extra_headers" in params:
        return websockets.connect(url, extra_headers=headers, **kwargs)

    # Last-resort: try without headers (will fail auth, but avoids crashing)
    logger.warning("websockets.connect doesn't accept extra/additional headers; attempting without headers")
    return websockets.connect(url, **kwargs)


class MongoDBTranscriptStorage:
    """
    MongoDB-based transcript storage for AI interviews.
    Stores transcripts directly in MongoDB database.
    """
    
    def __init__(self, interview_feedback_collection):
        """
        Initialize MongoDB storage.
        
        Args:
            interview_feedback_collection: MongoDB collection for interview feedback
        """
        self.collection = interview_feedback_collection
    
    async def save_message(self, session_id: str, role: str, text: str):
        """
        Save a message to the conversation transcript in MongoDB.
        
        Args:
            session_id: Unique session identifier (feedback_id)
            role: Either "user" or "assistant"
            text: The message text
        """
        try:
            # Get IST time (UTC + 5:30)
            from datetime import timedelta
            ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
            
            message = {
                "timestamp": ist_now.isoformat(),  # Store in IST
                "role": role,
                "text": text
            }
            
            # Append message to transcript array
            await self.collection.update_one(
                {"feedback_id": session_id},
                {
                    "$push": {"transcript": message},
                    "$set": {"updated_at": ist_now}  # Store in IST
                }
            )
            logger.info(f"💾 Saved {role} message to MongoDB: {text[:50]}...")
            
        except Exception as e:
            logger.error(f"Error saving message to MongoDB: {e}")
    
    async def get_conversation(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve a conversation by session ID.
        
        Args:
            session_id: Unique session identifier
            
        Returns:
            Conversation dictionary or None if not found
        """
        try:
            return await self.collection.find_one({"feedback_id": session_id})
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None


class AIInterviewAgent:
    """
    AI Interview Agent for real-time voice conversations.
    Handles bidirectional audio streaming with OpenAI Realtime API.
    Manages WebSocket connections and audio processing for AI interviews.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        system_message: Optional[str] = None,
        voice: str = "shimmer",
        model: str = "gpt-4o-realtime-preview-2024-12-17",
        audio_format: str = "pcm16",
        temperature: float = 0.6,
        storage: Optional[MongoDBTranscriptStorage] = None,
        jd_content: str = "",
        resume_content: str = "",
    ):
        """
        Initialize the AI interview agent.
        
        Args:
            openai_api_key: OpenAI API key
            system_message: System prompt/knowledge base
            voice: Voice to use (shimmer, echo, alloy, fable, onyx, nova)
            model: OpenAI model to use
            audio_format: Audio format (pcm16 for browser)
            temperature: AI temperature (0.0-1.0)
            storage: MongoDB transcript storage instance
            jd_content: Job Description text content
            resume_content: Resume text content
        """
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var.")
        
        self.jd_content = jd_content
        self.resume_content = resume_content
        
        self.system_message = system_message or get_system_message(
            phone_number="",
            jd_content=jd_content,
            resume_content=resume_content
        )
        
        # Log system message length for debugging
        print(f"📝 System message length: {len(self.system_message)} chars")
        print(f"📝 System message preview: {self.system_message[:200]}...")
        self.voice = voice
        self.model = model
        self.audio_format = audio_format
        self.temperature = temperature
        self.storage = storage
        
        self.session_id: Optional[str] = None
        self.termination_event = asyncio.Event()
        self.start_time: Optional[float] = None
        self.stream_sid: Optional[str] = None
        self.openai_ws = None
        self.client_ws = None
        self.user_speaking = False
        self.current_ai_response = ""
        self.last_user_transcript = ""
    
    async def connect(
        self,
        client_websocket,
        session_id: Optional[str] = None,
        timeout_seconds: int = 1200  # 20 minutes for interview
    ):
        """
        Connect and start the AI interview conversation.
        
        Args:
            client_websocket: WebSocket connection from client
            session_id: Unique session identifier
            timeout_seconds: Session timeout in seconds
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.start_time = time.time()
        self.client_ws = client_websocket
        self.user_speaking = False
        
        print(f"🎯 Inside connect() method for session: {self.session_id}")
        
        try:
            print(f"✅ Client WebSocket ready. Session: {self.session_id}")
            logger.info(f"✅ Client WebSocket ready. Session: {self.session_id}")
            
            # Connect to OpenAI Realtime API
            print(f"🔗 Attempting to connect to OpenAI Realtime API...")
            logger.info(f"🔗 Connecting to OpenAI Realtime API...")
            logger.info(f"   Model: {self.model}")
            logger.info(f"   API Key: {self.openai_api_key[:20]}...")
            print(f"   Model: {self.model}")
            print(f"   API Key: {self.openai_api_key[:20]}...")
            
            openai_url = f"wss://api.openai.com/v1/realtime?model={self.model}"
            openai_headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "OpenAI-Beta": "realtime=v1",
            }

            async with _websockets_connect_with_headers(
                openai_url,
                headers=openai_headers,
                ping_interval=20,
                ping_timeout=10,
            ) as openai_ws:
                self.openai_ws = openai_ws
                logger.info(f"✅ Connected to OpenAI Realtime API")
                
                try:
                    # Configure OpenAI session
                    await self._setup_session(openai_ws)
                    
                    # Trigger AI to start speaking
                    await self._trigger_ai_start(openai_ws)
                    
                    # Start timeout checker
                    asyncio.create_task(self._check_timeout(client_websocket, openai_ws, timeout_seconds))
                    
                    # Start bidirectional audio streaming
                    results = await asyncio.gather(
                        self._receive_from_client(client_websocket, openai_ws),
                        self._send_to_client(client_websocket, openai_ws),
                        return_exceptions=True
                    )
                    
                    # Log any exceptions
                    for i, result in enumerate(results):
                        if isinstance(result, Exception):
                            task_name = ["receive_from_client", "send_to_client"][i]
                            logger.error(f"Task {task_name} raised exception: {result}", exc_info=result)
                    
                except websockets.exceptions.ConnectionClosed:
                    logger.info(f"OpenAI WebSocket closed. Session: {self.session_id}")
                except Exception as e:
                    logger.error(f"Error in conversation: {e}", exc_info=True)
                finally:
                    await self._cleanup(client_websocket, openai_ws)
                    
        except websockets.exceptions.InvalidStatusCode as e:
            error_msg = f"❌ Failed to connect to OpenAI Realtime API: Status {e.status_code}"
            logger.error(error_msg)
            logger.error(f"Response headers: {e.headers}")
            print(error_msg)
            if hasattr(client_websocket, 'close'):
                await client_websocket.close(code=1011, reason="OpenAI connection failed - check API key")
        except websockets.exceptions.InvalidURI as e:
            error_msg = f"❌ Invalid OpenAI WebSocket URI: {e}"
            logger.error(error_msg)
            print(error_msg)
            if hasattr(client_websocket, 'close'):
                await client_websocket.close(code=1011, reason="Invalid OpenAI URI")
        except Exception as e:
            error_msg = f"❌ Critical error in AI interview: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            print(error_msg)
            import traceback
            traceback.print_exc()
            if hasattr(client_websocket, 'close'):
                try:
                    await client_websocket.close(code=1011, reason=str(e)[:100])
                except:
                    pass
    
    async def _setup_session(self, openai_ws):
        """Configure OpenAI session with system message and audio settings."""
        session_update = {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "instructions": self.system_message,
                "voice": self.voice,
                "input_audio_format": self.audio_format,
                "output_audio_format": self.audio_format,
                "input_audio_transcription": {
                    "model": "whisper-1",
                    "language": "en"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.4,
                    "prefix_padding_ms": 200,
                    "silence_duration_ms": 200
                },
                "temperature": self.temperature
            }
        }
        await openai_ws.send(json.dumps(session_update))
        logger.info("✅ Session configured with English-only transcription")
        await asyncio.sleep(1)
    
    async def _trigger_ai_start(self, openai_ws):
        """Trigger AI to start speaking naturally."""
        trigger_request = {
            "type": "response.create",
            "response": {
                "modalities": ["text", "audio"]
            }
        }
        await openai_ws.send(json.dumps(trigger_request))
        logger.info(f"✅ AI triggered to start interview")
    
    async def _receive_from_client(self, client_ws, openai_ws):
        """Receive audio from browser client and send to OpenAI."""
        logger.info(f"🎤 Starting to receive from client. Session: {self.session_id}")
        while not self.termination_event.is_set():
            try:
                # Check if WebSocket is still connected
                if hasattr(client_ws, 'client_state'):
                    if client_ws.client_state.name != "CONNECTED":
                        logger.info(f"Client WebSocket not connected. Session: {self.session_id}")
                        self.termination_event.set()
                        break
                
                # Receive message from client
                message = await client_ws.receive_text()
                
                if not message:
                    continue
                
                data = json.loads(message)
                
                # Handle audio media
                if data.get('event') == 'media':
                    audio_payload = None
                    if 'media' in data and 'payload' in data['media']:
                        audio_payload = data['media']['payload']
                    elif 'payload' in data:
                        audio_payload = data['payload']
                    
                    if audio_payload:
                        # Send to OpenAI
                        audio_append = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_payload
                        }
                        await openai_ws.send(json.dumps(audio_append))
                        
                        if self.start_time:
                            self.start_time = time.time()  # Reset timeout
                
                # Handle stream start
                elif data.get('event') == 'start':
                    self.stream_sid = data.get('streamSid', 'browser')
                    self.start_time = time.time()
                    logger.info(f"📡 Stream started: {self.stream_sid}")
                
                # Handle end session (user clicked End Interview)
                elif data.get('event') == 'end_session':
                    logger.info(f"🛑 Client requested end session. Session: {self.session_id}")
                    self.termination_event.set()
                    break
                
            except Exception as e:
                error_str = str(e).lower()
                if "disconnect" in error_str or "closed" in error_str:
                    logger.info(f"Client WebSocket disconnected. Session: {self.session_id}")
                    self.termination_event.set()
                    break
                logger.warning(f"Error in receive_from_client: {e}")
                await asyncio.sleep(0.1)
        
        logger.info(f"🎤 Stopped receiving from client. Session: {self.session_id}")
    
    async def _send_to_client(self, client_ws, openai_ws):
        """Receive audio/text from OpenAI and send to client."""
        try:
            async for openai_message in openai_ws:
                if self.termination_event.is_set():
                    break
                
                try:
                    response = json.loads(openai_message)
                    event_type = response.get('type', '')
                    
                    # Log all OpenAI events for debugging
                    if event_type not in ['response.audio.delta', 'input_audio_buffer.append']:
                        logger.info(f"📨 OpenAI event: {event_type}")
                    
                    # Log error details
                    if event_type == 'error':
                        error_details = response.get('error', {})
                        logger.error(f"❌ OpenAI Error: {error_details}")
                    
                    # Reset timeout on activity
                    if self.start_time:
                        self.start_time = time.time()
                    
                    # Handle user speech detection
                    if event_type == "input_audio_buffer.speech_started":
                        logger.info("User started speaking - STOPPING AI")
                        self.user_speaking = True
                        try:
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))
                            await self._send_message_to_client(client_ws, {"event": "stop_audio"})
                            await self._send_message_to_client(client_ws, {"event": "ai_response_complete"})
                        except Exception as e:
                            logger.warning(f"Error canceling response: {e}")
                    
                    # Reset flag when user stops speaking
                    if event_type == "input_audio_buffer.speech_stopped":
                        logger.info("User stopped speaking")
                        self.user_speaking = False
                    
                    # Handle user transcription
                    if event_type == 'conversation.item.input_audio_transcription.completed':
                        transcription = self._extract_transcription(response)
                        if transcription:
                            await self._handle_user_speech(transcription)
                            await self._send_message_to_client(client_ws, {
                                "event": "transcription",
                                "text": transcription
                            })
                    
                    # Handle conversation items
                    if event_type == 'conversation.item.created':
                        item = response.get('item', {})
                        role = item.get('role', '')
                        
                        if role == 'user':
                            transcription = self._extract_transcription_from_item(item)
                            if transcription and transcription != self.last_user_transcript:
                                self.last_user_transcript = transcription
                                await self._handle_user_speech(transcription)
                                await self._send_message_to_client(client_ws, {
                                    "event": "transcription",
                                    "text": transcription
                                })
                        
                        elif role == 'assistant':
                            ai_text = self._extract_text_from_item(item)
                            if ai_text:
                                await self._handle_ai_speech(ai_text)
                                await self._send_message_to_client(client_ws, {
                                    "event": "ai_response",
                                    "text": ai_text
                                })
                    
                    # Handle audio transcript deltas
                    if event_type == 'response.audio_transcript.delta':
                        delta = response.get('delta', '')
                        if delta:
                            self.current_ai_response += delta
                            await self._send_message_to_client(client_ws, {
                                "event": "ai_response",
                                "text": delta
                            })
                    
                    # Handle audio transcript done
                    if event_type == 'response.audio_transcript.done':
                        complete_transcript = response.get('transcript', '')
                        if complete_transcript:
                            await self._handle_ai_speech(complete_transcript)
                            await self._send_message_to_client(client_ws, {
                                "event": "ai_response_complete"
                            })
                            self.current_ai_response = ""
                    
                    # Handle response done
                    if event_type == 'response.done':
                        logger.info(f"📋 response.done - current_ai_response: '{self.current_ai_response}'")
                        logger.info(f"📋 response.done - response keys: {list(response.keys())}")
                        
                        # Check if we got any output
                        resp_obj = response.get('response', {})
                        output = resp_obj.get('output', [])
                        status = resp_obj.get('status', 'unknown')
                        status_details = resp_obj.get('status_details', {})
                        
                        logger.info(f"📋 response output items: {len(output)}")
                        logger.info(f"📋 response status: {status}")
                        if status_details:
                            logger.info(f"📋 response status_details: {status_details}")
                        
                        # Log the full response object for debugging
                        print(f"🔍 Full response.done object: {json.dumps(resp_obj, indent=2)}")
                        
                        if self.current_ai_response:
                            await self._handle_ai_speech(self.current_ai_response)
                        await self._send_message_to_client(client_ws, {
                            "event": "ai_response_complete"
                        })
                        self.current_ai_response = ""
                    
                    # Handle audio output
                    if event_type == 'response.audio.delta' and response.get('delta'):
                        if not self.user_speaking:
                            await self._send_audio_to_client(client_ws, response['delta'])
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    logger.error(f"Error processing OpenAI message: {e}")
        
        except Exception as e:
            logger.error(f"Error in send_to_client: {e}", exc_info=True)
            self.termination_event.set()
        finally:
            logger.info(f"🔊 Stopped receiving from OpenAI. Session: {self.session_id}")
    
    async def _send_audio_to_client(self, client_ws, audio_delta: str):
        """Send audio data to client WebSocket."""
        try:
            audio_payload = base64.b64encode(base64.b64decode(audio_delta)).decode('utf-8')
            audio_message = {
                "event": "media",
                "media": {"payload": audio_payload}
            }
            if self.stream_sid:
                audio_message["streamSid"] = self.stream_sid
            await self._send_message_to_client(client_ws, audio_message)
        except Exception as e:
            logger.error(f"Error sending audio to client: {e}")
    
    async def _send_message_to_client(self, client_ws, message: dict):
        """Send any message to client WebSocket."""
        try:
            if hasattr(client_ws, 'send_json'):
                if hasattr(client_ws, 'client_state'):
                    if client_ws.client_state.name == "CONNECTED":
                        await client_ws.send_json(message)
                else:
                    await client_ws.send_json(message)
            elif hasattr(client_ws, 'send'):
                await client_ws.send(json.dumps(message))
        except Exception as e:
            logger.debug(f"Error sending message to client: {e}")
    
    def _extract_transcription(self, response: Dict) -> Optional[str]:
        """Extract transcription text from various response formats."""
        transcription = (
            response.get('transcript') or
            response.get('text') or
            response.get('transcription', '')
        )
        if not transcription and 'item' in response:
            item = response.get('item', {})
            transcription = item.get('transcript') or item.get('text') or item.get('transcription', '')
        if not transcription and 'input_audio_transcription' in response:
            trans_obj = response.get('input_audio_transcription', {})
            transcription = trans_obj.get('transcript') or trans_obj.get('text', '')
        return transcription if transcription else None
    
    def _extract_transcription_from_item(self, item: Dict) -> Optional[str]:
        """Extract transcription from conversation item."""
        if 'input_audio_transcription' in item:
            trans_obj = item.get('input_audio_transcription', {})
            return trans_obj.get('transcript') or trans_obj.get('text', '')
        return None
    
    def _extract_text_from_item(self, item: Dict) -> Optional[str]:
        """Extract text content from conversation item."""
        if 'content' in item:
            full_text = ""
            for content_item in item.get('content', []):
                if content_item.get('type') == 'text':
                    text = content_item.get('text', '')
                    if text:
                        full_text += text + " "
            return full_text.strip() if full_text.strip() else None
        return None
    
    async def _handle_user_speech(self, transcription: str):
        """Handle user speech transcription."""
        logger.info(f"📝 User said: {transcription}")
        if self.storage and self.session_id:
            await self.storage.save_message(self.session_id, "user", transcription)
    
    async def _handle_ai_speech(self, text: str):
        """Handle AI speech text."""
        logger.info(f"🤖 AI said: {text}")
        if self.storage and self.session_id:
            await self.storage.save_message(self.session_id, "assistant", text)
    
    async def _check_timeout(self, client_ws, openai_ws, timeout_seconds: int):
        """Check for session timeout."""
        try:
            while not self.termination_event.is_set():
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    if elapsed > timeout_seconds:
                        logger.info(f"⏱️ Session timeout after {timeout_seconds}s. Session: {self.session_id}")
                        self.termination_event.set()
                        break
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error in timeout checker: {e}")
    
    async def _cleanup(self, client_ws, openai_ws):
        """Cleanup connections."""
        try:
            if openai_ws:
                try:
                    await openai_ws.close()
                except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
                    pass
            if hasattr(client_ws, 'close'):
                if hasattr(client_ws, 'client_state') and client_ws.client_state.name == "CONNECTED":
                    await client_ws.close()
            logger.info(f"✅ Cleaned up connections. Session: {self.session_id}")
        except Exception as e:
            logger.debug(f"Error during cleanup: {e}")
    
    def stop(self):
        """Manually stop the conversation."""
        self.termination_event.set()
        logger.info(f"🛑 Agent stopped. Session: {self.session_id}")
