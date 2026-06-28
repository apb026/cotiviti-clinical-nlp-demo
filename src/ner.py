"""
ner.py
Clinical Named Entity Recognition using a free, local Hugging Face model
(d4data/biomedical-ner-all). No paid APIs, runs fully on CPU.
"""

from typing import Dict, List
from transformers import pipeline

NER_MODEL_NAME = "d4data/biomedical-ner-all"


class ClinicalNER:
    """Wraps a Hugging Face biomedical NER pipeline for clinical entity extraction."""

    def __init__(self, model_name: str = NER_MODEL_NAME):
        self.model_name = model_name
        self.pipeline = pipeline(
            "ner",
            model=self.model_name,
            tokenizer=self.model_name,
            aggregation_strategy="simple",
        )

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Run NER on text and group entity strings by their predicted label.
        Returns: {label: [unique entity strings]}
        """
        raw_entities = self.pipeline(text)

        grouped: Dict[str, List[str]] = {}
        for ent in raw_entities:
            label = ent["entity_group"]
            word = ent["word"].strip()
            if not word:
                continue
            grouped.setdefault(label, [])
            if word.lower() not in [w.lower() for w in grouped[label]]:
                grouped[label].append(word)

        return grouped

    def extract_flat(self, text: str) -> List[Dict]:
        """Return the raw list of entity dicts (label, word, score, position)."""
        return self.pipeline(text)
