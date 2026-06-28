"""
risk.py
Simple, transparent, rule-based risk scoring built on top of NER output.
No black-box model required - this keeps the logic auditable, which matters
in healthcare contexts where explainability is critical.
"""

from typing import Dict, List

HIGH_RISK_TERMS = [
    "stemi", "myocardial infarction", "stroke", "ischemic stroke",
    "hemorrhage", "sepsis", "respiratory failure", "cardiac arrest",
]

MEDIUM_RISK_TERMS = [
    "pneumonia", "diabetes", "hypertension", "chronic kidney disease",
    "fracture", "dyslipidemia", "cancer", "chemotherapy",
]


def score_from_entities(entities: Dict[str, List[str]]) -> Dict:
    """
    Scan all extracted entity strings (regardless of label) for known
    high/medium risk clinical terms and compute a simple 0-100 risk score.
    """
    all_terms: List[str] = []
    for label, words in entities.items():
        all_terms.extend(words)

    text_blob = " ".join(all_terms).lower()

    score = 0
    matched_factors = []

    for term in HIGH_RISK_TERMS:
        if term in text_blob:
            score += 35
            matched_factors.append(f"High-risk finding: {term}")

    for term in MEDIUM_RISK_TERMS:
        if term in text_blob:
            score += 15
            matched_factors.append(f"Medium-risk finding: {term}")

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
