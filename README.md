# Minimal local uploader

This small FastAPI app accepts a multipart file upload and uploads the file to a Google Cloud Storage bucket (default `subtitle-input`).

Prerequisites:

- Python 3.9+
- A Google Cloud service account JSON key with permission to write to the bucket.

Setup (Windows cmd.exe):

1. Create a venv and install deps:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Set environment variables (replace path and bucket name):

```bat
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account.json
set GCP_BUCKET_NAME=subtitle-input
```

3. Run the server:

```bat
python new.py
```

4. Test with curl (from git bash or use PowerShell equivalent). From cmd.exe you can use `curl` on modern Windows:

```bat
curl -v -X POST http://127.0.0.1:8000/upload -F "file=@C:\path\to\video.mp4"
```

What the app returns:

- `gcs_path`: gs://bucket/object
- `bucket`: the bucket used
- `object`: the object path inside the bucket

Notes:

- The code uses Application Default Credentials. For local testing set `GOOGLE_APPLICATION_CREDENTIALS` to a service account JSON.
- If you prefer signed or public URLs, you can generate them using the Storage client after upload.
