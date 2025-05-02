# Text Summarizer

A Python project for extractive and abstractive text summarization, website/article summarization, evaluation metrics, keyword extraction, and a Streamlit web app.

## Features
- **Extractive Summarization**: Summarize text using algorithms like TF-IDF and TextRank.
- **Abstractive Summarization**: Generate summaries using transformer models (e.g., BART).
- **Website Summarization**: Extract and summarize content from URLs.
- **Evaluation Metrics**: Measure summary quality using compression ratio, content retention, and ROUGE scores.
- **Keyword Extraction**: Identify the most relevant keywords from the text.
- **Streamlit Web Interface**: User-friendly interface for summarization tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/text-summarizer.git
   cd text-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Run the Web App

1. Start the FastAPI server:
   ```bash
   python Backend/webserver.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

### Use the Python API

Import the `TextSummarizer` class from `src/summarizer.py` and use it in your Python scripts:

```python
from src.summarizer import TextSummarizer

summarizer = TextSummarizer(method='textrank')
text = "Your input text here."
summary = summarizer.summarize(text, ratio=0.3)
print(summary)
```

## Project Structure

- `src/summarizer.py`: Main summarizer logic.
- `Backend/webserver.py`: FastAPI backend for the web app.
- `Frontend/`: Contains the HTML, CSS, and JavaScript for the web interface.
- `requirements.txt`: List of dependencies.
- `Readme.md`: Project documentation.

## Updates
- Removed language-specific logic for simplicity.
- Focused on English text summarization.
- Improved modular design and error handling.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---
