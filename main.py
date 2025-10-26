from pathlib import Path
import os
import uuid
import time
import asyncio
import subprocess
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Enable logging output for all loggers
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import Request
import hmac
import hashlib

import httpx
import google
from google.cloud import storage
import firebase_admin
from firebase_admin import credentials, firestore, auth
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import base64
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./tmp_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "subtitlegen-input")
# Default to the full transcribe path used by your sample curl
WHISPER_ENDPOINT = os.getenv("WHISPER_ENDPOINT", "https://whisper-infer-gpu-627061442405.asia-southeast1.run.app/transcribe")

app = FastAPI(title="CapsAI Minimal Upload", version="0.0.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# ========= Firebase / Firestore =========
if not firebase_admin._apps:
    # For local development, use the JSON file in the workspace
    cert_path = os.getenv("FIREBASE_CERT_PATH", "caps-85254-firebase-adminsdk.json")
    if Path(cert_path).exists():
        cred = credentials.Certificate(cert_path)
        firebase_admin.initialize_app(cred)
    else:
        raise RuntimeError(f"Firebase credentials file not found at {cert_path}")
db = firestore.client()


# GCP storage functions are disabled for local/dev
def get_storage_client():
    return storage.Client()

async def upload_to_gcs(local_path: Path, dest_name: str) -> str:
    import logging
    logger = logging.getLogger("capsai.gcp.upload")
    try:
        logger.info(f"Starting GCP upload: local_path={local_path}, dest_name={dest_name}, bucket={GCP_BUCKET_NAME}")
        client = get_storage_client()
        bucket = client.bucket(GCP_BUCKET_NAME)
        blob = bucket.blob(dest_name)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, blob.upload_from_filename, str(local_path))
        logger.info(f"Upload to GCP bucket succeeded: gs://{bucket.name}/{dest_name}")
        # Return the GCP gs:// URL (not signed, not public)
        gcp_url = f"gs://{bucket.name}/{dest_name}"
        logger.info(f"Returning GCP URL: {gcp_url}")
        return gcp_url
    except Exception as e:
        logger.error(f"GCP upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GCP upload failed: {e}")
# ------------------ Auth helpers ------------------
async def verify_firebase_token(authorization: str = Header(None)):
    """Verify Firebase ID token from Authorization header and return uid and user doc."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Load user doc if exists
        user_doc = db.collection("users").document(uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        return {"uid": uid, "user": user_data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")

# Authenticated endpoint to stream video from GCP
@app.get("/api/download-video")
async def download_video(video_url: str, user=Depends(verify_firebase_token)):
    """Authenticated endpoint to stream video from GCP."""
    if not video_url.startswith("https://storage.googleapis.com/"):
        raise HTTPException(status_code=400, detail="Invalid video URL")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", video_url) as resp:
                resp.raise_for_status()
                content = b""
                async for chunk in resp.aiter_bytes():
                    content += chunk
                return FileResponse(
                    path=None,
                    content=content,
                    media_type="video/mp4",
                    headers={"Content-Disposition": "attachment; filename=video.mp4"}
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stream video: {e}")


async def forward_to_whisper(https_url: str, word_timestamps: bool = True, require_duration: bool = False, language: str = "en") -> dict:
    """Forward the HTTPS GCS URL to the Whisper /transcribe endpoint.

    Matches the sample curl payload:
    {
      "source_url": "https://storage.googleapis.com/...",
      "word_timestamps": true,
      "require_duration": false
    }
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {
            "source_url": https_url,
            "word_timestamps": bool(word_timestamps),
            "require_duration": bool(require_duration),
            "language": language
        }
        resp = await client.post(WHISPER_ENDPOINT, json=payload)
        resp.raise_for_status()
        try:
            return resp.json()
        except Exception:
            return {"text": resp.text}


# ---- Helpers needed by authenticated change-style ----
async def download_to_tmp(url: str, suffix: str = ".mp4") -> Path:
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    async for chunk in resp.aiter_bytes():
                        f.write(chunk)
        return tmp_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

def ffprobe_duration_seconds(path: Path) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        duration = float(out.decode().strip())
        if duration <= 0:
            cmd2 = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path)
            ]
            out2 = subprocess.check_output(cmd2, stderr=subprocess.STDOUT)
            duration = float(out2.decode().strip())
        return max(duration, 0.0)
    except Exception:
        return 0.0

async def get_video_duration_minutes(video_url: str) -> float:
    tmp = await download_to_tmp(video_url, ".bin")
    try:
        secs = ffprobe_duration_seconds(tmp)
        return secs / 60.0
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass

async def verify_firebase_token(authorization: str = Header(None)):
    """Verify Firebase ID token from Authorization header and return uid and user doc."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_doc = db.collection("users").document(uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        return {"uid": uid, "user": user_data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")


# ---------------- Cashfree / Payments helpers ----------------
def load_minutes_mapping():
    """Load mapping from environment var MINUTES_FOR_RUPEES as JSON or use defaults.

    Format example: '{"99":120, "499":700}' meaning 99 rupees -> 120 minutes.
    Values are rupees (not cents). If absent, a default mapping is used.
    """
    raw = os.getenv("MINUTES_FOR_RUPEES")
    if raw:
        try:
            mapping = json.loads(raw)
            # normalize keys to ints
            return {int(k): float(v) for k, v in mapping.items()}
        except Exception:
            pass
    # sensible defaults
    return {99: 120.0, 199: 300.0, 499: 900.0}


MINUTES_FOR_RUPEES = load_minutes_mapping()


def rupees_to_minutes(amount_rupees: float) -> float:
    """Convert a rupee amount to minutes using exact match or prorated fallback."""
    # try exact integer rupee match
    try:
        key = int(round(amount_rupees))
        if key in MINUTES_FOR_RUPEES:
            return float(MINUTES_FOR_RUPEES[key])
    except Exception:
        pass
    # fallback: prorate based on smallest defined package
    if not MINUTES_FOR_RUPEES:
        return 0.0
    # choose smallest package by rupees
    smallest_rupee = min(MINUTES_FOR_RUPEES.keys())
    smallest_minutes = MINUTES_FOR_RUPEES[smallest_rupee]
    # prorate linearly
    return round((amount_rupees / float(smallest_rupee)) * float(smallest_minutes), 1)


def record_purchase(uid: str, order_id: str, amount_rupees: float, minutes: float, status: str = "created"):
    """Store a purchase doc in Firestore under purchases/{order_id}"""
    doc = {
        "uid": uid,
        "order_id": order_id,
        "amount_rupees": float(amount_rupees),
        "minutes": float(minutes),
        "status": status,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    db.collection("purchases").document(order_id).set(doc)


@app.post("/create-purchase")
async def create_purchase(amount_rupees: float = Form(...), user=Depends(verify_firebase_token)):
    """Create a purchase and (optionally) create a Cashfree order.

    Requires Firebase auth; UID is taken from the verified token. If Cashfree env vars are
    configured, an order is created via the Cashfree API and its checkout URL is returned.
    Otherwise a mocked checkout URL is returned for local testing.
    """
    uid = user.get("uid")
    if not uid or amount_rupees <= 0:
        raise HTTPException(status_code=400, detail="Invalid parameters")

    order_id = f"order_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    minutes = rupees_to_minutes(amount_rupees)
    record_purchase(uid, order_id, amount_rupees, minutes, status="created")

    # If Cashfree is configured, create a real order and return its checkout URL
    CASHFREE_API_URL = os.getenv("CASHFREE_API_URL")
    CASHFREE_CLIENT_ID = os.getenv("CASHFREE_CLIENT_ID")
    CASHFREE_CLIENT_SECRET = os.getenv("CASHFREE_CLIENT_SECRET")

    # Prefer sandbox Cashfree /pg/orders flow using client id/secret if present (as in your app.py)
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            cf_resp = await client.post(
                "https://sandbox.cashfree.com/pg/orders",
                json={
                    "order_amount": float(amount_rupees),
                    "order_currency": "INR",
                    "order_id": order_id,
                    "customer_details": {
                        "customer_id": uid,
                        "customer_name": os.getenv("CASHFREE_CUSTOMER_NAME", ""),
                        "customer_email": os.getenv("CASHFREE_CUSTOMER_EMAIL", ""),
                        "customer_phone": os.getenv("CASHFREE_CUSTOMER_PHONE", "9999999999"),
                    },
                },
                headers={
                    "x-client-id": os.getenv("CASHFREE_APPID"),
                    "x-client-secret": os.getenv("CASHFREE_SECRETKEY"),
                    "x-api-version": "2023-08-01",
                },
            )
            cf_resp.raise_for_status()
            order_json = cf_resp.json()
            # update purchase with raw cashfree response
            db.collection("purchases").document(order_id).update({"cashfree_response": order_json})
            # Try to extract a checkout/payment URL
            checkout_url = (
                order_json.get("payment_link")
                or order_json.get("paymentLink")
                or order_json.get("payment_url")
                or (order_json.get("order") or {}).get("payment_link")
            )
            if checkout_url:
                return {"order_id": order_id, "checkout_url": checkout_url, "minutes_expected": minutes}
    except Exception as e:
        # Log and continue to fallback mocked URL so purchases are still recorded
        print(f"Cashfree order creation failed (sandbox): {e}")

    # fallback mocked checkout URL for local testing
    checkout_url = os.getenv("CASHFREE_MOCK_CHECKOUT") or f"https://pay.example.com/checkout/{order_id}"
    return {"order_id": order_id, "checkout_url": checkout_url, "minutes_expected": minutes}


@app.post("/cashfree/webhook")
async def cashfree_webhook(request: Request):
    """Handle Cashfree webhook notifications.

    Cashfree will POST payment status. For security, validate a shared secret header CASHFREE_WEBHOOK_SECRET
    and ensure idempotent crediting of minutes.
    Expected body (example): {"order_id": "order_xxx", "status": "PAID", "amount": 99.0}
    """
    secret = os.getenv("CASHFREE_WEBHOOK_SECRET")
    header_secret = request.headers.get("X-Cashfree-Secret")
    if secret and header_secret != secret:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    payload = await request.json()
    order_id = payload.get("order_id") or payload.get("orderId")
    status = payload.get("status") or payload.get("payment_status")
    amount = payload.get("amount") or payload.get("amount_rupees") or payload.get("amountInRupees")

    if not order_id or not status:
        raise HTTPException(status_code=400, detail="Missing order_id or status")

    # Lookup purchase
    purchase_ref = db.collection("purchases").document(order_id)
    purchase_doc = purchase_ref.get()
    if not purchase_doc.exists:
        # Create a record if it didn't exist (incoming webhook for external-created order)
        # Determine minutes from payload amount if present
        amount_val = float(amount) if amount else 0.0
        minutes = rupees_to_minutes(amount_val)
        record_purchase(uid=payload.get("uid", "unknown"), order_id=order_id, amount_rupees=amount_val, minutes=minutes, status=status)
        purchase_doc = purchase_ref.get()

    purchase = purchase_doc.to_dict()

    # If already marked credited, return 200 (idempotent)
    if purchase.get("status") == "credited":
        return {"ok": True, "message": "already credited"}

    # Only credit on success statuses (Cashfree uses PAID or SUCCESS depending on config)
    success_states = {"PAID", "SUCCESS", "COMPLETED"}
    if str(status).upper() in success_states:
        uid = purchase.get("uid")
        if not uid or uid == "unknown":
            # If UID missing, we'll mark the purchase and skip credit — admin can reconcile
            purchase_ref.update({"status": "paid_no_uid"})
            return {"ok": True, "message": "paid but no uid to credit"}

        minutes = float(purchase.get("minutes", 0))
        # Credit minutes atomically using transaction
        def txn_update(txn):
            user_ref = db.collection("users").document(uid)
            snapshot = user_ref.get(transactions=txn)
            current = float(snapshot.to_dict().get("videomins", 0)) if snapshot.exists else 0.0
            new_total = current + float(minutes)
            txn.update(user_ref, {"videomins": new_total})
            txn.update(purchase_ref, {"status": "credited", "credited_at": firestore.SERVER_TIMESTAMP})

        try:
            db.run_transaction(txn_update)
        except Exception:
            # Fallback non-transactional update if transactions fail for environments without them
            user_ref = db.collection("users").document(uid)
            user_doc = user_ref.get()
            current = float(user_doc.to_dict().get("videomins", 0)) if user_doc.exists else 0.0
            user_ref.set({"videomins": current + float(minutes)}, merge=True)
            purchase_ref.update({"status": "credited", "credited_at": firestore.SERVER_TIMESTAMP})

        return {"ok": True, "message": "credited", "uid": uid, "minutes": minutes}

    # Non-success statuses: mark paid/failure accordingly
    purchase_ref.update({"status": str(status).lower()})
    return {"ok": True, "message": "status recorded", "status": status}



#########################################################################################################



@app.post("/api/process-video")
async def process_video(
    video: UploadFile = File(...),
    language: str = Form("en"),
    style: str = Form(None),
    # user=Depends(verify_firebase_token)
):
    import time
    start_time = time.time()
    if not video.filename:
        return JSONResponse(status_code=400, content={
            "success": False,
            "message": "No video file provided",
            "data": None
        })

    suffix = Path(video.filename).suffix or ".mp4"
    local_name = f"{int(time.time())}_{uuid.uuid4().hex}{suffix}"
    local_path = UPLOAD_DIR / local_name

    try:
        with open(local_path, "wb") as f:
            content = await video.read()
            f.write(content)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"Failed saving uploaded file: {e}",
            "data": None
        })

    # Upload to GCP bucket and get the signed URL
    gcp_url = None
    try:
        dest_name = local_name
        gcp_url = await upload_to_gcs(local_path, dest_name)
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "message": f"GCP upload failed: {e}",
            "data": None
        })

    # Optionally remove local file after upload
    try:
        local_path.unlink(missing_ok=True)
    except Exception:
        pass

    https_url = gcp_url

    # Forward to Whisper inference endpoint and include its response
    whisper_result = None
    try:
        whisper_result = await forward_to_whisper(https_url, word_timestamps=True, require_duration=True, language=language)
    except Exception as e:
        whisper_result = {"error": str(e)}

    # Convert Whisper JSON response to SRT (if possible)
    def seconds_to_srt_timestamp(s: float) -> str:
        ms = int(round(s * 1000))
        hours = ms // 3600000
        ms -= hours * 3600000
        minutes = ms // 60000
        ms -= minutes * 60000
        seconds = ms // 1000
        ms -= seconds * 1000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

    def transcription_json_to_srt(trans_json: dict) -> str:
        MAX_WORDS_PER_LINE = 7
        MIN_GAP_MS = 200
        srt_lines = []
        subtitle_index = 1
        current_words = []
        # Always extract segments from trans_json['transcription']['segments'] if present
        segments = None
        if isinstance(trans_json, dict):
            transcription = trans_json.get("transcription")
            if isinstance(transcription, dict) and "segments" in transcription:
                segments = transcription["segments"]
            elif "segments" in trans_json:
                segments = trans_json["segments"]
        if not segments:
            return ""
        def flush_words():
            nonlocal current_words, subtitle_index
            if not current_words:
                return
            start = float(current_words[0].get("start", 0))
            end = float(current_words[-1].get("end", start))
            text = " ".join(word.get("word", "").strip() for word in current_words)
            text = text.strip()
            if text:
                srt_lines.append(str(subtitle_index))
                srt_lines.append(f"{seconds_to_srt_timestamp(start)} --> {seconds_to_srt_timestamp(end)}")
                srt_lines.append(text)
                srt_lines.append("")
                subtitle_index += 1
            current_words = []
        for seg in segments:
            words = seg.get("words", [])
            if not words:
                continue
            for word in words:
                if not word.get("word", "").strip():
                    continue
                if current_words and word.get("start"):
                    prev_end = float(current_words[-1].get("end", 0)) * 1000
                    curr_start = float(word.get("start", 0)) * 1000
                    if curr_start - prev_end >= MIN_GAP_MS:
                        flush_words()
                current_words.append(word)
                text = word.get("word", "").strip()
                if len(current_words) >= MAX_WORDS_PER_LINE or text[-1:] in ".!?":
                    flush_words()
        if current_words:
            flush_words()
        return "\n".join(srt_lines)

    # Calculate processing time
    processing_time = round(time.time() - start_time, 2)

    # Prepare response data
    response_data = {
        "videoUrl": https_url,
        "subtitles": whisper_result.get("transcription", {}).get("segments", []) if isinstance(whisper_result, dict) else [],
        "transcription": whisper_result.get("transcription", {}).get("text", "") if isinstance(whisper_result, dict) else "",
        "processingTime": f"{processing_time}s"
    }

    return JSONResponse(status_code=200, content={
        "success": True,
        "message": "Video processed successfully",
        "data": response_data
    })


from pydantic import BaseModel
from typing import List
from fastapi import Depends, Header


class FontStyle(BaseModel):
    color: str = "#FFFFFF"
    fontSize: int = 28
    fontFamily: str = "Poppins, sans-serif"
    fontWeight: int = 700



from typing import Dict

class SubtitleLine(BaseModel):
    timeStart: str
    timeEnd: str
    value: str

class ChangeStyleRequest(BaseModel):
    videoSrc: str
    subtitles: List[SubtitleLine]
    font: Dict
    videoResolution: str


def parse_srt_timestamp(timestamp: str) -> str:
    """Keep SRT timestamp as is since Remotion accepts the same format."""
    return timestamp


def srt_to_remotion_subtitles(srt_text: str) -> List[dict]:
    """Convert SRT format to Remotion subtitle format."""
    subtitles = []
    current_subtitle = {}
    
    lines = srt_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            if current_subtitle:
                subtitles.append(current_subtitle)
                current_subtitle = {}
            continue
            
        if ' --> ' in line:
            start, end = line.split(' --> ')
            current_subtitle['timeStart'] = parse_srt_timestamp(start)
            current_subtitle['timeEnd'] = parse_srt_timestamp(end)
        elif not line[0].isdigit():  # Skip subtitle numbers
            current_subtitle['value'] = line
    
    # Add the last subtitle if exists
    if current_subtitle:
        subtitles.append(current_subtitle)
    
    return subtitles


# ----- Email / verification models & helpers (copied from app.py) -----
class EmailRequest(BaseModel):
    email: str
    userName: str

class VerificationCodeRequest(BaseModel):
    email: str

class VerifyCodeRequest(BaseModel):
    email: str
    code: int
    uid: str | None = None


def create_gmail_service():
    """Create and return Gmail API service using OAuth2 credentials"""
    try:
        client_id = os.getenv('GMAIL_CLIENT_ID')
        client_secret = os.getenv('GMAIL_CLIENT_SECRET')
        refresh_token = os.getenv('GMAIL_REFRESH_TOKEN')
        
        if not all([client_id, client_secret, refresh_token]):
            raise ValueError("Missing Gmail API credentials. Ensure GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN are set.")
        
        # Create credentials using OAuth2 tokens
        creds = Credentials(
            None,  # No access token initially
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=['https://www.googleapis.com/auth/gmail.send']
        )
        
        # Build the Gmail service
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Gmail service: {e}")

async def send_email(to_email: str, subject: str, html_content: str):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = os.getenv('EMAIL_FROM', 'ai.editor@capsai.co')
        msg['To'] = to_email
        msg.attach(MIMEText(html_content, 'html'))

        # Create Gmail API service
        service = create_gmail_service()
        
        # Encode the message
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
        message = {'raw': raw}
        
        # Send the email
        service.users().messages().send(userId='me', body=message).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Email sending failed: {e}")


def generate_order_id() -> str:
    return hashlib.sha256(os.urandom(16)).hexdigest()[:12]


@app.post("/api/send-welcome-email")
async def send_welcome_email(request: EmailRequest):
    # Compose a concise HTML welcome email, using the user's name if provided
    user_name = getattr(request, 'userName', None) or getattr(request, 'name', None) or ''
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Welcome to Capsai</title>
        <style>
            body {{ background: linear-gradient(135deg, #e0e7ff 0%, #f9fafb 100%); font-family: 'Inter', Arial, sans-serif; margin:0; padding:0; }}
            .container {{ max-width:520px; margin:2rem auto; background:#fff; border-radius:16px; box-shadow:0 4px 24px #dbeafe; padding:2.5rem 2rem; }}
            .logo {{ text-align:center; margin-bottom:1.5rem; }}
            .logo img {{ width:64px; }}
            h2 {{ color:#0066cc; font-size:2rem; margin-bottom:0.5rem; text-align:center; }}
            .welcome {{ font-size:1.1rem; color:#333; text-align:center; margin-bottom:1.5rem; }}
            .cta {{ display:block; width:fit-content; margin:2rem auto 0 auto; background:#0066cc; color:#fff; padding:0.75rem 2rem; border-radius:8px; text-decoration:none; font-weight:600; font-size:1rem; box-shadow:0 2px 8px #e0e7ff; transition:background 0.2s; }}
            .cta:hover {{ background:#005bb5; }}
            .footer {{ text-align:center; color:#888; margin-top:2rem; font-size:0.95rem; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <img src="https://capsai.co/favicon.ico" alt="Capsai Logo" />
            </div>
            <h2>Welcome to Capsai{f', {user_name}' if user_name else ''}!</h2>
            <div class="welcome">
                We're thrilled to have you join our creative community.<br>
                <b>Capsai</b> helps you turn your videos into magic with fast, accurate subtitles and beautiful styles.<br>
                <br>
                <span style="color:#28a745;font-weight:500;">Ready to get started?</span>
            </div>
            <a href="https://capsai.co/workspace" class="cta">Explore Your Workspace</a>
            <div class="footer">
                Need help? <a href="https://capsai.co/workspace/help" style="color:#0066cc;text-decoration:underline;">Support Center</a><br>
                &copy; 2025 Capsai
            </div>
        </div>
    </body>
    </html>
    """
    try:
        await send_email(request.email, "Welcome to Capsai", html_content)
        return {"message": "Email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/creds-refuel")
async def creds_refuel(request: EmailRequest):
    user_name = getattr(request, 'userName', None) or getattr(request, 'name', None) or ''
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Refuel Your Minutes</title>
        <style>
            .container {{ max-width:500px; margin:auto; background:#fff; border-radius:10px; box-shadow:0 2px 8px #eee; padding:2rem; font-family:Arial,sans-serif; }}
            .header {{ color:#28a745; font-size:2rem; margin-bottom:1rem; }}
            .cta-btn {{ display:inline-block; background:#0066cc; color:#fff; padding:0.75rem 1.5rem; border-radius:5px; text-decoration:none; font-weight:bold; margin-top:1.5rem; }}
            .highlight {{ color:#ff9800; font-weight:bold; }}
        </style>
    </head>
    <body style='background:#f4f8fb;'>
        <div class='container'>
            <div class='header'>Time to Refuel, {user_name if user_name else 'Capsai Creator'}!</div>
            <p>Your creative journey deserves more minutes. <span class='highlight'>Plans start at just ₹29</span> — get more video minutes instantly and keep creating magic!</p>
            <a href='https://capsai.co/workspace/pricing' class='cta-btn'>Refuel Now</a>
            <p style='margin-top:2rem;'>Need help? Reply to this email or visit our <a href='https://capsai.co/workspace/help'>Support Center</a>.</p>
            <hr style='margin:2rem 0;'>
            <small style='color:#888;'>Capsai &copy; 2025</small>
        </div>
    </body>
    </html>
    """
    try:
        await send_email(request.email, "Refuel Your Minutes-Plans Starting at ₹29", html_content)
        return {"message": "Email sent successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/sendVerificationCode-email-auth")
async def send_verification_code(request: VerificationCodeRequest):
    code = random.randint(1000, 9999)
    try:
        db.collection("verificationCodes").document(request.email).set({
            "code": code, "expiresAt": int(time.time() * 1000) + 60000
        })
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Your Verification Code</title>
        </head>
        <body>
            <h1>Your Verification Code</h1>
            <p>Here is your verification code: <strong>{code}</strong></p>
            <p>This code will expire in 1 minute.</p>
        </body>
        </html>
        """
        await send_email(request.email, "Your Verification Code", html_content)
        return {"message": "Verification code sent"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/verifyCode-email-auth")
async def verify_code(request: VerifyCodeRequest):
    try:
        doc = db.collection("verificationCodes").document(request.email).get()
        if not doc.exists:
            raise HTTPException(status_code=400, detail="Invalid or expired code")
        data = doc.to_dict()
        now_ms = int(time.time() * 1000)
        if data["code"] != request.code or now_ms > data["expiresAt"]:
            raise HTTPException(status_code=400, detail="Invalid or expired code")
        db.collection("verificationCodes").document(request.email).delete()
        token = auth.create_custom_token(request.email)
        # auth.create_custom_token returns bytes; decode for JSON
        return {"token": token.decode("utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/payment")
async def create_payment(order_amount: float, customer_id: str, customer_name: str, customer_email: str):
    try:
        order_id = generate_order_id()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                "https://sandbox.cashfree.com/pg/orders",
                json={
                    "order_amount": order_amount,
                    "order_currency": "INR",
                    "order_id": order_id,
                    "customer_details": {
                        "customer_id": customer_id,
                        "customer_name": customer_name,
                        "customer_email": customer_email,
                        "customer_phone": "9999999999",
                    },
                },
                headers={
                    "x-client-id": os.getenv("CASHFREE_APPID"),
                    "x-client-secret": os.getenv("CASHFREE_SECRETKEY"),
                    "x-api-version": "2023-08-01",
                },
            )
            r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class VerifyPaymentRequest(BaseModel):
    orderId: str


@app.post("/api/verify")
async def verify_payment(request: VerifyPaymentRequest):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(
                f"https://sandbox.cashfree.com/pg/orders/{request.orderId}/payments",
                headers={
                    "x-client-id": os.getenv("CASHFREE_APPID"),
                    "x-client-secret": os.getenv("CASHFREE_SECRETKEY"),
                    "x-api-version": "2023-08-01",
                },
            )
            r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


REMOTION_ENDPOINT = os.getenv(
    "REMOTION_ENDPOINT",
    "https://remotion-backend-kyvjwe2urq-el.a.run.app/render"
)




# Refactored to match /api/change-style-remotion spec
from fastapi import Form, File, UploadFile
from typing import Optional
import json


@app.post("/api/change-style-remotion")
async def change_style(
    user=Depends(verify_firebase_token),
    videoUrl: str = Form(...),
    subtitles: str = Form(...),
    style: str = Form(...),
    watermarkConfig: Optional[str] = Form(None)
):
    """
    Accepts multipart form:
    - videoUrl: required string (GCP URL from previous step)
    - subtitles: required string (JSON array)
    - style: required string (JSON object)
    - watermarkConfig: optional string (JSON object)
    """
    import logging
    logger = logging.getLogger("capsai.change-style-remotion")
    # Parse subtitles and style
    try:
        subtitles_json = json.loads(subtitles)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid subtitles JSON: {e}", "data": None})
    try:
        style_json = json.loads(style)
    except Exception as e:
        return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid style JSON: {e}", "data": None})
    watermark_json = None
    if watermarkConfig:
        try:
            watermark_json = json.loads(watermarkConfig)
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "message": f"Invalid watermarkConfig JSON: {e}", "data": None})

    # Prepare payload for Remotion backend
    remotion_payload = {
        "videoUrl": videoUrl,
        "subtitles": subtitles_json,
        "style": style_json,
    }
    if watermark_json:
        remotion_payload["watermarkConfig"] = watermark_json

    # Call Remotion backend
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            remotion_resp = await client.post(REMOTION_ENDPOINT, json=remotion_payload)
            remotion_resp.raise_for_status()
            result = remotion_resp.json()
    except Exception as e:
        logger.error(f"Remotion backend call failed: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "message": f"Remotion backend failed: {e}", "data": None})

    # Deduct video minutes if user is authenticated
    duration_minutes = 0.0
    render_video_url = result.get("video_url") or result.get("outputUrl")
    try:
        if render_video_url:
            duration_minutes = await get_video_duration_minutes(render_video_url)
        elif videoUrl:
            duration_minutes = await get_video_duration_minutes(videoUrl)
    except Exception:
        duration_minutes = 0.0

    uid = user["uid"]
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    if user_doc.exists:
        current_mins = float(user_doc.to_dict().get("videomins", 0))
    else:
        current_mins = 3.0
        user_ref.set({
            "email": uid,
            "image": "",
            "name": "",
            "username": uid.split('@')[0],
            "usertype": "free",
            "videomins": current_mins
        }, merge=True)
    remaining = max(0.0, round(current_mins - duration_minutes, 1))
    user_ref.set({"videomins": remaining}, merge=True)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "message": "Remotion video generated successfully",
            "data": result,
            "remaining_minutes": remaining
        }
    )


@app.post("/api/add-minutes")
async def add_minutes(uid: str, minutes: float, api_key: str = Header(None)):
    """Admin endpoint to add minutes to a user. Protect with ADMIN_API_KEY env var."""
    if not api_key or api_key != os.getenv("ADMIN_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid admin API key")
    if not uid or minutes <= 0:
        raise HTTPException(status_code=400, detail="Invalid parameters")
    user_ref = db.collection("users").document(uid)
    user_doc = user_ref.get()
    current = float(user_doc.to_dict().get("videomins", 0)) if user_doc.exists else 0.0
    user_ref.set({"videomins": current + float(minutes)}, merge=True)
    return {"uid": uid, "videomins": current + float(minutes)}


# ------------------ Auth helpers ------------------
async def verify_firebase_token(authorization: str = Header(None)):
    """Verify Firebase ID token from Authorization header and return uid and user doc."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    id_token = authorization.split(" ", 1)[1]
    try:
        decoded = auth.verify_id_token(id_token)
        uid = decoded.get("uid")
        if not uid:
            raise HTTPException(status_code=401, detail="Invalid token")
        # Load user doc if exists
        user_doc = db.collection("users").document(uid).get()
        user_data = user_doc.to_dict() if user_doc.exists else {}
        return {"uid": uid, "user": user_data}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token verification failed: {e}")


@app.get("/api/me")
async def me(user=Depends(verify_firebase_token)):
    """Return basic user info and remaining minutes."""
    return {
        "uid": user["uid"],
        "user": user["user"],
        "videomins": user["user"].get("videomins", 0) if isinstance(user["user"], dict) else 0
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("new:app", port=int(os.getenv("PORT", 8080)))
