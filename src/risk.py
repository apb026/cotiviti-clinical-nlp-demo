"""
risk.py

Transparent, rule-based document review insights generated from
clinical entities extracted through the NER workflow.

The purpose is not clinical risk prediction, but to highlight
findings that may warrant additional review, validation, or
documentation attention while keeping the decision logic fully
auditable and explainable.
"""

from typing import Dict, List

HIGH_PRIORITY_TERMS = [
    "stemi", "myocardial infarction", "stroke", "ischemic stroke",
    "hemorrhage", "sepsis", "respiratory failure", "cardiac arrest",
]

MEDIUM_PRIORITY_TERMS = [
    "pneumonia", "diabetes", "hypertension", "chronic kidney disease",
    "fracture", "dyslipidemia", "cancer", "chemotherapy",
]


def score_from_entities(entities: Dict[str, List[str]]) -> Dict:
    """
    Analyze extracted clinical entities and generate a simple,
    transparent review-priority score based on predefined findings
    that may benefit from additional review.
    """
    all_terms: List[str] = []
    for label, words in entities.items():
        all_terms.extend(words)

    text_blob = " ".join(all_terms).lower()

    score = 0
    matched_factors = []

    for term in HIGH_PRIORITY_TERMS:
        if term in text_blob:
            score += 35
            matched_factors.append(f"High-priority review finding: {term}")

    for term in MEDIUM_PRIORITY_TERMS:
        if term in text_blob:
            score += 15
            matched_factors.append(f"Medium-priority review finding: {term}")

    score = min(score, 100)
    if score >= 50:
        level = "HIGH"
    elif score >= 20:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "risk_score": score,
        "risk_level": level,
        "risk_factors": matched_factors,
    }
