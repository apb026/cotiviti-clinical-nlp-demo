"""
app.py
Clinical NLP Proof of Concept - Streamlit App

100% free, open-source, local stack:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (Hugging Face)
- Vector search: FAISS (faiss-cpu)
- NER: d4data/biomedical-ner-all (Hugging Face)
- Summarization: sshleifer/distilbart-cnn-12-6 (Hugging Face)
- OCR: Tesseract via pytesseract

No paid APIs. No cloud infrastructure. No paid datasets.
All clinical text in data/clinical_notes.json is synthetic.
"""

import sys
import os

sys.path.append(os.path.dirname(__file__))

import streamlit as st
from PIL import Image

from src.retrieval import ClinicalRetriever
from src.ner import ClinicalNER
from src.summarizer import ClinicalSummarizer
from src.ocr import extract_text_from_image, check_tesseract_installed
from src.risk import score_from_entities

st.set_page_config(
    page_title="Clinical Policy Intelligence Assistant",
    layout="wide"
)


# ---------------------------------------------------------------------------
# Cached model loaders (so models load once per session, not per interaction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model + building FAISS index...")
def load_retriever():
    return ClinicalRetriever()


@st.cache_resource(show_spinner="Loading biomedical NER model...")
def load_ner():
    return ClinicalNER()


@st.cache_resource(show_spinner="Loading summarization model...")
def load_summarizer():
    return ClinicalSummarizer()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Clinical Policy Intelligence Assistant")
st.sidebar.markdown(
    """
### Technical Components

- Semantic Retrieval
- Clinical Information Extraction
- Document Summarization
- Document Digitization (OCR)
- Content Review Insights
"""
)

tesseract_ok = check_tesseract_installed()
if tesseract_ok:
    st.sidebar.success("Tesseract OCR detected ✓")
else:
    st.sidebar.warning("Tesseract OCR not found — OCR tab will not work until installed.")

st.title("🩺 Clinical Policy Intelligence Assistant")
st.caption(
    "Healthcare document analysis, information extraction, semantic retrieval, "
    "document summarization, and document processing workflows."
)
st.markdown(
    """
> **Healthcare Content Management Workflows Demonstrated**
>
> **Clinical Content Search** enables rapid retrieval of relevant healthcare information using semantic similarity rather than keyword matching.
>
> **Clinical Information Extraction** identifies key clinical concepts, diagnoses, procedures, and medications that may support documentation review and content organization.
>
> **Document Summarization** reduces the effort required to review lengthy healthcare documents by generating concise summaries.
>
> **Document Digitization** converts scanned or image-based documents into machine-readable text for downstream processing and analysis.
>
> **Document Review Insights** demonstrates how extracted information can be analyzed to support review workflows and highlight content that may warrant additional attention.
"""
)

tab_search, tab_ner, tab_summary, tab_ocr, tab_risk = st.tabs(
    ["🔍 Semantic Search", "🏷️ Entity Extraction", "📝 Summarization", "📷 OCR", "⚠️ Risk Scoring"]
)

# ---------------------------------------------------------------------------
# TAB 1: Semantic Search (FAISS + embeddings)
# ---------------------------------------------------------------------------
with tab_search:
    st.subheader("Semantic Document Retrieval")
    st.markdown(
    "Retrieve relevant healthcare content using semantic similarity. "
    "This workflow demonstrates how clinical documents, policies, and "
    "other healthcare content can be searched based on meaning rather "
    "than exact keyword matches."
)

    retriever = load_retriever()

    query = st.text_input(
        "Enter a clinical query",
        value="cardiac event requiring intervention",
    )
    top_k = st.slider("Number of results", 1, 5, 3)

    if st.button("Search", key="search_btn"):
        results = retriever.search(query, top_k=top_k)
        for i, doc in enumerate(results, 1):
            with st.expander(
                f"#{i} — {doc['type']} ({doc['specialty']}) | similarity: {doc['similarity_score']:.3f}"
            ):
                st.write(doc["content"])

# ---------------------------------------------------------------------------
# TAB 2: Entity Extraction (Hugging Face NER)
# ---------------------------------------------------------------------------
with tab_ner:
    st.subheader("Clinical Entity Extraction")
    st.markdown(
    "Identifies clinically relevant concepts such as diagnoses, medications, "
    "and procedures to support document review, content organization, and "
    "healthcare information management workflows."
    )

    retriever = load_retriever()
    doc_options = {f"{d['id']} — {d['type']}": d["content"] for d in retriever.documents}
    choice = st.selectbox("Pick a sample document, or paste your own text below", list(doc_options.keys()))

    default_text = doc_options[choice]
    text_input = st.text_area("Text to analyze", value=default_text, height=180)

    if st.button("Extract Entities"):
        ner_model = load_ner()
        with st.spinner("Running NER..."):
            entities = ner_model.extract(text_input)

        if not entities:
            st.info("No entities detected.")
        else:
            for label, words in entities.items():
                st.markdown(f"**{label}**: {', '.join(words)}")

        st.session_state["last_entities"] = entities
        st.session_state["last_text"] = text_input

# ---------------------------------------------------------------------------
# TAB 3: Summarization (Hugging Face DistilBART)
# ---------------------------------------------------------------------------
with tab_summary:
    st.subheader("Clinical Document Summarization")
    st.markdown(
    "Generates concise summaries of healthcare documents to reduce manual "
    "review effort and improve access to key information."
    )

    retriever = load_retriever()
    doc_options2 = {f"{d['id']} — {d['type']}": d["content"] for d in retriever.documents}
    choice2 = st.selectbox(
        "Pick a sample document, or paste your own text below",
        list(doc_options2.keys()),
        key="summary_select",
    )

    default_text2 = doc_options2[choice2]
    if (
        "last_summary_choice" not in st.session_state
        or st.session_state["last_summary_choice"] != choice2
    ):
        st.session_state["summary_text"] = default_text2
        st.session_state["last_summary_choice"] = choice2
    text_input2 = st.text_area(
    "Text to summarize",
    value=default_text2,
    height=180,
    # key=f"summary_text_{choice2}"
    )

    col1, col2 = st.columns(2)
    max_len = col1.slider("Max summary length", 20, 150, 80)
    min_len = col2.slider("Min summary length", 5, 60, 20)

    if st.button("Summarize"):
        summarizer = load_summarizer()
        with st.spinner("Summarizing..."):
            summary = summarizer.summarize(text_input2, max_length=max_len, min_length=min_len)
        st.success("Summary:")
        st.write(summary)

# ---------------------------------------------------------------------------
# TAB 4: OCR (Tesseract)
# ---------------------------------------------------------------------------
with tab_ocr:
    st.subheader("OCR: Scanned Document Text Extraction")
    st.markdown(
    "Convert scanned or image-based healthcare documents into machine-readable "
    "text that can be searched, analyzed, and summarized through downstream NLP workflows."
    )

    if not tesseract_ok:
        st.error(
            "Tesseract OCR binary not found on this system. Install it with:\n\n"
            "`sudo apt-get install tesseract-ocr` (Linux) or `brew install tesseract` (macOS)."
        )

    uploaded_file = st.file_uploader("Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Run OCR") and tesseract_ok:
            with st.spinner("Extracting text..."):
                extracted_text = extract_text_from_image(image)
            st.success("Extracted text:")
            st.text_area("OCR Output", value=extracted_text, height=200)
            st.session_state["ocr_text"] = extracted_text
            st.info("Tip: copy this text into the Entity Extraction or Summarization tabs.")

# ---------------------------------------------------------------------------
# TAB 5: Risk Scoring (rule-based, built on NER output)
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Transparent, Rule-Based Risk Scoring")
    st.markdown(
    "Demonstrates how extracted clinical information can be analyzed to "
    "support document review workflows and highlight content that may "
    "benefit from additional review or validation."
)

    retriever = load_retriever()
    doc_options3 = {f"{d['id']} — {d['type']}": d["content"] for d in retriever.documents}
    choice3 = st.selectbox(
        "Pick a sample document, or paste your own text below",
        list(doc_options3.keys()),
        key="risk_select",
    )
    default_text3 = doc_options3[choice3]

    if (
        "last_risk_choice" not in st.session_state
        or st.session_state["last_risk_choice"] != choice3
    ):
        st.session_state["risk_text"] = default_text3
        st.session_state["last_risk_choice"] = choice3

    text_input3 = st.text_area(
        "Text to assess",
        key="risk_text",
        height=180,
    )

    if st.button("Compute Risk Score"):
        ner_model = load_ner()
        with st.spinner("Extracting entities and scoring risk..."):
            entities = ner_model.extract(text_input3)
            risk = score_from_entities(entities)

        level_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[risk["risk_level"]]
        st.markdown(f"### {level_color} Risk Level: {risk['risk_level']} ({risk['risk_score']}/100)")

        if risk["risk_factors"]:
            st.markdown("**Risk factors detected:**")
            for factor in risk["risk_factors"]:
                st.write(f"- {factor}")
        else:
            st.write("No specific risk factors detected by the rule set.")

        with st.expander("View extracted entities used for scoring"):
            st.json(entities)

st.divider()
st.caption(
    "Built with Streamlit, Hugging Face Transformers, Sentence-Transformers, "
    "FAISS, and Tesseract OCR — all free and open-source. No data leaves your machine."
)
