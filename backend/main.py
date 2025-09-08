# server.py
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests, trafilatura, time, os, logging
from dotenv import load_dotenv

# load .env in local dev
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# read secrets from env (must set on Railway / locally)
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

if not BRAVE_API_KEY:
    logger.warning("BRAVE_API_KEY not set. Requests to Brave API will fail.")
if not HF_API_TOKEN:
    logger.warning("HF_API_TOKEN not set. HF summarization will fail.")

HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# CORS: allow localhost for dev and vercel preview domains (regex)
# CORS: allow localhost for dev and your Vercel app only
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",             # local dev
        "https://froont-h1cg.vercel.app",    # your deployed frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# optional request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"REQ {request.method} {request.url} ORIGIN={request.headers.get('origin')}")
    response = await call_next(request)
    logger.info(f"RESP {response.status_code} {request.url}")
    return response


def summarize_via_hf(text: str, max_retries: int = 2, backoff: float = 1.0) -> str:
    if not HF_API_TOKEN:
        return "Summary error: HF_API_TOKEN not configured"
    if not text or not text.strip():
        return None

    payload = {
        "inputs": text,
        "parameters": {"min_length": 30, "max_length": 130, "do_sample": False},
    }

    attempt = 0
    while attempt <= max_retries:
        try:
            resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
            if resp.status_code == 429:
                time.sleep(backoff * (attempt + 1))
                attempt += 1
                continue
            if resp.status_code >= 400:
                return f"Summary error (status {resp.status_code}): {resp.text[:500]}"
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict) and "summary_text" in first:
                    return first["summary_text"]
                if isinstance(first, str):
                    return first
            if isinstance(data, dict):
                if "error" in data:
                    return f"Summary error: {data.get('error')}"
                if "summary_text" in data:
                    return data["summary_text"]
            return str(data)[:1000]
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(backoff * (attempt + 1))
                attempt += 1
                continue
            return f"Summary request failed: {str(e)}"
    return "Summary unavailable after retries"


@app.get("/search_summary")
def search_and_summarize(
    q: str = Query(..., description="Search query"),
    country: str = Query("US", description="Country code (e.g., US, BE, IN)"),
    ui_lang: str = Query("en-US", description="UI language (e.g., en-US, fr-FR, hi-IN)"),
):
    if not BRAVE_API_KEY:
        raise HTTPException(status_code=500, detail="BRAVE_API_KEY not configured")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": q, "count": 7, "country": country, "ui_lang": ui_lang}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
    except Exception as e:
        logger.exception("Error calling Brave API")
        raise HTTPException(status_code=502, detail=f"Brave API request failed: {e}")

    if response.status_code != 200:
        logger.error("Brave API returned non-200: %s %s", response.status_code, response.text[:400])
        raise HTTPException(status_code=502, detail=f"Brave API: {response.status_code}")

    data = response.json()
    results = []
    url_summaries = []

    for item in data.get("web", {}).get("results", [])[:7]:
        url_item = item.get("url")
        content_preview = None
        summary = None

        try:
            downloaded = trafilatura.fetch_url(url_item)
            if downloaded:
                extracted = trafilatura.extract(downloaded)
                if extracted:
                    content_preview = extracted[:900]
                    summary = summarize_via_hf(content_preview)
                    if summary and not (summary.startswith("Summary error") or summary.startswith("Summary request failed")):
                        url_summaries.append(summary)
        except Exception as e:
            logger.exception("Error extracting URL: %s", url_item)
            content_preview = f"Error extracting: {str(e)}"

        results.append(
            {
                "title": item.get("title"),
                "url": url_item,
                "description": item.get("description"),
                "content_preview": content_preview,
                "summary": summary,
            }
        )

    combined_summary = " ".join(url_summaries) if url_summaries else None

    return {
        "query": q,
        "country": country,
        "ui_lang": ui_lang,
        "results": results,
        "combined_summary": combined_summary,
    }
