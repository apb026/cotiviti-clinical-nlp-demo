"""
summarizer.py
Extractive/abstractive summarization using a free, local, lightweight
Hugging Face model (sshleifer/distilbart-cnn-12-6). No paid APIs.
"""

from transformers import pipeline

SUMMARIZER_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"


class ClinicalSummarizer:
    """Wraps a Hugging Face summarization pipeline."""

    def __init__(self, model_name: str = SUMMARIZER_MODEL_NAME):
        self.model_name = model_name
        self.pipeline = pipeline("summarization", model=self.model_name)

    def summarize(self, text: str, max_length: int = 80, min_length: int = 20) -> str:
        # DistilBART has a max input length; truncate defensively for the demo
        text = text[:3000]
        result = self.pipeline(
            text, max_length=max_length, min_length=min_length, do_sample=False
        )
        return result[0]["summary_text"]
