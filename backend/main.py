from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests, trafilatura, time, os
from dotenv import load_dotenv

# Load .env
load_dotenv()

app = FastAPI()

# Read from environment
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

HF_API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}", "Accept": "application/json"}

# CORS config (same as before)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def summarize_via_hf(text: str, max_retries: int = 2, backoff: float = 1.0) -> str:
    """
    Call Hugging Face Inference API to summarize `text`.
    Returns the summary string, or an error message string (prefixed) on failure.
    Retries on transient errors (429, network).
    """
    if not text or not text.strip():
        return None

    # You can tune parameters (min_length/max_length) here if you wish
    payload = {
        "inputs": text,
        "parameters": {"min_length": 30, "max_length": 130, "do_sample": False},
    }

    attempt = 0
    while attempt <= max_retries:
        try:
            resp = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)
            # rate-limited
            if resp.status_code == 429:
                # exponential backoff-ish
                time.sleep(backoff * (attempt + 1))
                attempt += 1
                continue

            if resp.status_code >= 400:
                # return an error string so caller can include it in the response
                return f"Summary error (status {resp.status_code}): {resp.text[:500]}"

            data = resp.json()

            # HF returns a list like: [{"summary_text":"..."}]
            if isinstance(data, list) and len(data) > 0:
                first = data[0]
                if isinstance(first, dict) and "summary_text" in first:
                    return first["summary_text"]
                if isinstance(first, str):
                    return first

            # some endpoints may return a dict with 'error'
            if isinstance(data, dict):
                if "error" in data:
                    return f"Summary error: {data.get('error')}"
                if "summary_text" in data:
                    return data["summary_text"]

            # fallback: stringify the response
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
    """
    Search Brave Web API with optional country and ui_lang, fetch pages,
    extract text with trafilatura, and summarize each page using the Hugging Face Inference API.
    Returns results list and a combined_summary.
    """
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {"q": q, "count": 7, "country": country, "ui_lang": ui_lang}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return {"error": response.status_code, "message": response.text}

    data = response.json()
    results = []
    url_summaries = []

    for item in data.get("web", {}).get("results", [])[:7]:
        url_item = item.get("url")
        content_preview = None
        summary = None

        try:
            # fetch and extract page content
            downloaded = trafilatura.fetch_url(url_item)
            if downloaded:
                extracted = trafilatura.extract(downloaded)
                if extracted:
                    # Truncate to 900 chars for faster summarization (tweak as needed)
                    content_preview = extracted[:900]

                    # Call HF Inference API to summarize the content_preview
                    summary = summarize_via_hf(content_preview)

                    # Only add summary to combined list if it's not an HF error string
                    if summary and not (summary.startswith("Summary error") or summary.startswith("Summary request failed")):
                        url_summaries.append(summary)
        except Exception as e:
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

    # Combine all URL summaries into a single text
    combined_summary = " ".join(url_summaries) if url_summaries else None

    return {
        "query": q,
        "country": country,
        "ui_lang": ui_lang,
        "results": results,
        "combined_summary": combined_summary,
    }
