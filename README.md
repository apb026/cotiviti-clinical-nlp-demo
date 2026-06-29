# Clinical Policy Intelligence Assistant

A Streamlit application demonstrating Clinical Natural Language Processing
capabilities for healthcare content analysis, information retrieval,
document understanding, and workflow support.

The application showcases semantic search, entity extraction,
summarization, OCR-based document processing, and policy-oriented
content exploration.

## Technical Architecture

| Component | Technology |
|-----------|------------|
| UI | [Streamlit](https://streamlit.io) | Apache 2.0 |
| Embeddings | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Hugging Face) | Apache 2.0 |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) | MIT |
| Clinical NER | [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) (Hugging Face) | Apache 2.0 |
| Summarization | [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) (Hugging Face) | Apache 2.0 |
| OCR | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) via `pytesseract` | Apache 2.0 |
| Data | Demonstration clinical documents used to showcase retrieval, entity extraction, summarization, and document-processing workflows |


---

## What's Inside

```
clinical-policy-intelligence-assistant/
├── .devcontainer/
│   ├── devcontainer.json   # GitHub Codespaces container config (port 8501, 4 CPU / 8GB)
│   └── setup.sh             # Installs Tesseract + Python deps + pre-caches models
├── .streamlit/
│   └── config.toml          # Headless server config so Codespaces can forward the port
├── app.py                  # Streamlit app (5 tabs)
├── requirements.txt
├── data/
│   └── clinical_notes.json # 8 synthetic clinical notes (no real PHI)
└── src/
    ├── retrieval.py         # FAISS + sentence-transformers semantic search
    ├── ner.py                # Hugging Face biomedical NER
    ├── summarizer.py         # Hugging Face DistilBART summarization
    ├── ocr.py                # Tesseract OCR wrapper
    └── risk.py               # Transparent, rule-based risk scoring
```

### App Tabs (and how each maps to the report's three use cases)

1. **Clinical Content Search** — type a clinical query, get back the most relevant
   synthetic notes ranked by cosine similarity (FAISS over MiniLM embeddings).
   *Maps to: payer policy comparison* — the same retrieval mechanics apply
   directly to finding relevant policy language by meaning, not keyword.
2. **EClinical Information Extraction** — runs a real biomedical NER model over any note
   (or your own pasted text) and groups entities by type.
   *Maps to: coding productivity* — surfacing diagnoses/procedures/meds
   without a coder reading the full narrative.
3. **Document Summarization** — condenses a clinical note using a local DistilBART
   model.
   *Maps to: coding productivity* — shorter notes to review per chart.
4. **Document Digitization** — upload a photo/scan of a document; Tesseract extracts the text,
   which can then be analyzed using the Clinical Information Extraction or Document Summarization workflows.
   *Underlies all three use cases* — most source documents start as scans
   or faxes, not clean text.
5. 5. **Document Review Insights** — analyzes extracted clinical findings and generates transparent review recommendations based on predefined rules. The workflow highlights information that may warrant additional review, validation, or documentation attention while keeping the decision logic fully explainable.

   *Maps to: documentation validation* — demonstrates an auditable review workflow that surfaces potentially important findings and supports manual review processes.

---

## Extending This Project

- Swap `all-MiniLM-L6-v2` for a domain-specific embedding model like
  `pritamdeka/S-PubMedBert-MS-MARCO` for better clinical semantic search.
- Replace the rule-based review workflow with a small scikit-learn classifier
  trained on structured features extracted from NER output.
- Add `pdf2image` to support PDF uploads in the OCR tab (scanned PDFs →
  images → Tesseract).
- Persist the FAISS index to disk (`faiss.write_index`) instead of rebuilding
  it on every app restart, once the corpus grows beyond the demo size.

---
