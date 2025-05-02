# Import necessary libraries for text processing and summarization
import os
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from rouge_score import rouge_scorer
from typing import List, Dict, Any
from googletrans import Translator

# Set environment variable to disable TensorFlow warnings
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Download necessary NLTK data
nltk.download(["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger"])


class TextSummarizer:
    """
    Main summarizer class supporting extractive, abstractive,
    and website summarization.
    """
    def __init__(
        self,
        method: str = 'textrank',
        min_summary_length: int = 3
    ):
        """
        Initialize the summarizer with the specified method and minimum \
        summary length.
        """
        self.method = method
        self.min_summary_length = min_summary_length
        self.language = 'english'
        try:
            self.stop_words = set(stopwords.words('english'))
        except OSError:
            self.stop_words = set()

    def preprocess(self, text: str) -> str:
        """
        Preprocess the input text by removing punctuation, converting to
        lowercase, and filtering out stopwords and irrelevant parts of speech.
        """
        text = re.sub(r'[^\w\s]', '', text)
        words = word_tokenize(text.lower(), language='english')
        try:
            pos_tags = nltk.pos_tag(words, lang='english')
        except Exception:
            pos_tags = nltk.pos_tag(words)
        keep_tags = {
            'NN', 'NNS', 'NNP', 'NNPS', 'JJ', 'JJR', 'JJS',
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
        }
        filtered_words = [
            word for word, tag in pos_tags
            if tag in keep_tags and word not in self.stop_words
        ]
        return ' '.join(filtered_words)

    def _tfidf_summarize(self, sentences: List[str]) -> np.ndarray:
        """
        Perform extractive summarization using TF-IDF.
        """
        preprocessed = [self.preprocess(sent) for sent in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
        return np.array(tfidf_matrix.sum(axis=1)).flatten()

    def _textrank_summarize(self, sentences: List[str]) -> np.ndarray:
        """
        Perform extractive summarization using TextRank.
        """
        preprocessed = [self.preprocess(sent) for sent in sentences]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
        sim_matrix = cosine_similarity(tfidf_matrix)
        np.fill_diagonal(sim_matrix, 0)
        scores = np.ones(len(sentences))
        for _ in range(20):
            scores = 0.15 + 0.85 * np.dot(sim_matrix, scores)
        return scores

    def summarize(
        self,
        text: str,
        ratio: float = 0.3,
        abstractive: bool = False,
        min_length: int = 30,
        max_length: int = 130
    ) -> str:
        """
        Summarize the input text using the specified method
        (extractive or abstractive).
        """
        # Translate to English if not English
        translator = Translator()
        try:
            text = translator.translate(
                text, src='auto', dest='english'
            ).text
        except Exception as e:
            raise ValueError(f"Translation to English failed: {e}")

        if abstractive:
            summary = self.summarize_abstractive(text, min_length, max_length)
        else:
            sentences = sent_tokenize(text, language='english')
            if len(sentences) <= self.min_summary_length:
                summary = text
            else:
                num_sentences = max(
                    int(len(sentences) * ratio), self.min_summary_length
                )
                if self.method == 'tfidf':
                    scores = self._tfidf_summarize(sentences)
                else:
                    scores = self._textrank_summarize(sentences)
                ranked = np.argsort(scores)[::-1][:num_sentences]
                ranked.sort()
                summary = ' '.join([sentences[i] for i in ranked])

        # Translate back to original language if needed
        try:
            summary = translator.translate(
                summary, src='english', dest='auto'
            ).text
        except Exception as e:
            raise ValueError(f"Translation back to original language failed: {e}")

        return summary

    def summarize_abstractive(
        self,
        text: str,
        min_length: int = 30,
        max_length: int = 130
    ) -> str:
        """
        Perform abstractive summarization using a transformer model.
        """
        summarizer = pipeline('summarization', model='facebook/bart-large-cnn')
        result = summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            truncation=True
        )
        return result[0]['summary_text']

    def summarize_url(self, url: str, **kwargs) -> str:
        """
        Fetch and summarize the content of a website URL.
        """
        article = self.fetch_website_text(url)
        return self.summarize(article, **kwargs)

    def fetch_website_text(self, url: str) -> str:
        """
        Extract the main content from a website URL.
        """
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'noscript']):
            tag.decompose()
        paragraphs = [p.get_text() for p in soup.find_all('p')]
        return '\n'.join(paragraphs).strip()

    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """
        Extract the top keywords from the input text using TF-IDF.
        """
        preprocessed = self.preprocess(text)
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([preprocessed])
        scores = tfidf_matrix.toarray()[0]
        feature_names = vectorizer.get_feature_names_out()
        top_indices = scores.argsort()[::-1][:top_k]
        return [feature_names[i] for i in top_indices if scores[i] > 0]

    def evaluate_summary(
        self,
        original: str,
        summary: str
    ) -> Dict[str, Any]:
        """
        Evaluate the quality of the summary using compression ratio
        and content retention.
        """
        orig_words = word_tokenize(original)
        summ_words = word_tokenize(summary)
        compression_ratio = 1 - (len(summ_words) / len(orig_words))
        content_retention = (
            len(set(summ_words) & set(orig_words)) / len(set(orig_words))
        )
        return {
            'compression_ratio': round(compression_ratio, 2),
            'content_retention': round(content_retention, 2),
            'original_length': len(orig_words),
            'summary_length': len(summ_words)
        }

    def evaluate_summary_rouge(
        self,
        original: str,
        summary: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of the summary using ROUGE scores.
        """
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        scores = scorer.score(original, summary)
        return {k: v.fmeasure for k, v in scores.items()}
