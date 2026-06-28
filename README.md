# Clinical NLP Proof of Concept (100% Free & Open-Source)

A Streamlit app demonstrating core clinical NLP capabilities using **only free,
open-source, local tools** — no paid APIs, no cloud infrastructure, no paid
healthcare datasets.

## Stack

| Capability | Tool | License |
|---|---|---|
| UI | [Streamlit](https://streamlit.io) | Apache 2.0 |
| Embeddings | [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Hugging Face) | Apache 2.0 |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`) | MIT |
| Clinical NER | [`d4data/biomedical-ner-all`](https://huggingface.co/d4data/biomedical-ner-all) (Hugging Face) | Apache 2.0 |
| Summarization | [`sshleifer/distilbart-cnn-12-6`](https://huggingface.co/sshleifer/distilbart-cnn-12-6) (Hugging Face) | Apache 2.0 |
| OCR | [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) via `pytesseract` | Apache 2.0 |
| Data | Synthetic clinical notes (`data/clinical_notes.json`), invented for this demo — **no real patient data, no PHI, no paid dataset** | N/A |

Everything runs **locally on CPU**. No internet connection is required once
the models have been downloaded the first time (Hugging Face caches them
locally in `~/.cache/huggingface`).

---

## Setup

### Option A: GitHub Codespaces (recommended — zero local setup)

1. Push this repo to GitHub.
2. Click **Code → Codespaces → Create codespace on main**.
3. Wait for the container to build (~3-5 min — it installs Tesseract OCR,
   Python deps, and pre-downloads the ~500MB of Hugging Face model weights
   automatically via `.devcontainer/setup.sh`).
4. Once it's ready, run:
   ```bash
   streamlit run app.py
   ```
5. Codespaces will pop up a "port forwarded" notification for port `8501` —
   click **Open in Browser**.

Everything here is free: Codespaces' free tier (60 core-hours/month on a
personal GitHub account) is enough to run and demo this app.

### Option B: Run locally

#### 1. Install the Tesseract OCR binary (system-level, not pip)

This is required for the OCR tab only — the rest of the app works without it.

```bash
# Ubuntu / Debian / WSL
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> First run will download ~500MB of model weights from Hugging Face
> (MiniLM, biomedical-ner-all, distilbart). This only happens once; they're
> cached locally afterward. A laptop CPU is sufficient — no GPU required.

### 4. Run the app

```bash
streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.

---

## What's Inside

```
clinical-nlp-poc/
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

1. **Semantic Search** — type a clinical query, get back the most relevant
   synthetic notes ranked by cosine similarity (FAISS over MiniLM embeddings).
   *Maps to: payer policy comparison* — the same retrieval mechanics apply
   directly to finding relevant policy language by meaning, not keyword.
2. **Entity Extraction** — runs a real biomedical NER model over any note
   (or your own pasted text) and groups entities by type.
   *Maps to: coding productivity* — surfacing diagnoses/procedures/meds
   without a coder reading the full narrative.
3. **Summarization** — condenses a clinical note using a local DistilBART
   model.
   *Maps to: coding productivity* — shorter notes to review per chart.
4. **OCR** — upload a photo/scan of a document; Tesseract extracts the text,
   which you can then feed into the NER or Summarization tabs.
   *Underlies all three use cases* — most source documents start as scans
   or faxes, not clean text.
5. **Risk Scoring** — a small, transparent, rule-based scorer that flags
   high/medium-risk findings from the extracted entities. Deliberately
   rule-based rather than a black-box model, so the logic is auditable.
   *Maps to: documentation validation* — an auditable flag for when
   extracted findings suggest higher acuity than what's reflected elsewhere.

---

## Why These Choices

- **No paid APIs / no cloud**: everything after the initial model download
  runs offline on your own machine.
- **FAISS over a hosted vector DB**: free, runs in-process, no server to
  manage — appropriate for a student project.
- **Hugging Face models over commercial LLM APIs**: free, open weights,
  no rate limits or billing.
- **Synthetic data**: avoids any need for licensed/paid clinical datasets or
  real PHI, while still being realistic enough to demonstrate the NLP
  pipeline meaningfully.
- **Rule-based risk scoring**: keeps that one component fully explainable —
  useful both as a teaching example and as a contrast to the "black box"
  feel of the NER/summarization models.

---

## Extending This Project

- Swap `all-MiniLM-L6-v2` for a domain-specific embedding model like
  `pritamdeka/S-PubMedBert-MS-MARCO` for better clinical semantic search.
- Replace the rule-based risk scorer with a small scikit-learn classifier
  trained on structured features extracted from NER output.
- Add `pdf2image` to support PDF uploads in the OCR tab (scanned PDFs →
  images → Tesseract).
- Persist the FAISS index to disk (`faiss.write_index`) instead of rebuilding
  it on every app restart, once the corpus grows beyond the demo size.

---

## Disclaimer

This is an educational proof-of-concept. All clinical text is synthetic and
invented for demonstration purposes. It is **not** intended for real clinical
use, diagnosis, or decision-making.
