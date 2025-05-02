# Import necessary libraries for FastAPI and summarization
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.summarizer import TextSummarizer
import uvicorn
import sys
import os

# Add the parent directory to the system path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '..')))

# Initialize the FastAPI app
app = FastAPI()

# Serve static files (frontend)
app.mount(
    "/static",
    StaticFiles(directory=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../Frontend"))),
    name="static"
)

# Initialize the summarizer instance
summarizer = TextSummarizer()


# Define the request model for summarization
class SummarizeRequest(BaseModel):
    text: str = ""  # Input text to summarize
    url: str = ""  # URL of the website to summarize
    method: str = "textrank"  # Summarization method (e.g., textrank, tfidf)
    ratio: float = 0.3  # Ratio of the summary length to the original text
    min_len: int = 3  # Minimum number of sentences in the summary
    min_abs: int = 30  # Minimum length for abstractive summaries
    max_abs: int = 130  # Maximum length for abstractive summaries
    top_k: int = 10  # Number of keywords to extract
    abstractive: bool = False  # Whether to use abstractive summarization


# Serve the frontend index.html file
@app.get("/", response_class=HTMLResponse)
def index():
    frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "../Frontend"))
    index_path = os.path.join(frontend_dir, "index.html")
    with open(index_path, encoding="utf-8") as f:
        return f.read()


# API endpoint for summarization
@app.post("/api/summarize")
def summarize(req: SummarizeRequest):
    # Initialize a new summarizer instance with the specified
    # method and minimum length
    summarizer = TextSummarizer(
        method=req.method,
        min_summary_length=req.min_len
    )

    # Fetch text from URL or use the provided text
    if req.url:
        article = summarizer.fetch_website_text(req.url)
        text = article
    else:
        text = req.text

    # Perform summarization based on the specified method
    if req.method.startswith("abstractive") or req.abstractive:
        summary = summarizer.summarize_abstractive(text,
                                                   min_length=req.min_abs,
                                                   max_length=req.max_abs)
    else:
        summary = summarizer.summarize(text, ratio=req.ratio)

    # Evaluate the summary and extract keywords
    evaluation = summarizer.evaluate_summary(text, summary)
    rouge = summarizer.evaluate_summary_rouge(text, summary)
    keywords = summarizer.extract_keywords(text, top_k=req.top_k)

    # Return the results as a JSON response
    return JSONResponse({
        "summary": summary,
        "evaluation": evaluation,
        "rouge": rouge,
        "keywords": keywords
    })


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    uvicorn.run("webserver:app", host="0.0.0.0", port=8000, reload=True)
