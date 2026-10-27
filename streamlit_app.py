
import os
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Document Sentiment Agent",
    page_icon="📄",
)

st.title("📄 Document Sentiment Agent")
st.write(
    "Upload a PDF or text document to analyze sentiment, "
    "identify evidence, and generate recommended actions."
)

# Read API credentials from Streamlit Secrets
try:
    for key in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "COMPANY_CONTEXT",
    ):
        if key in st.secrets:
            os.environ[key] = str(st.secrets[key])
except Exception:
    pass

if not os.environ.get("OPENROUTER_API_KEY"):
    st.error("Configure OPENROUTER_API_KEY in Streamlit Secrets.")
    st.stop()

if not os.environ.get("OPENROUTER_MODEL"):
    st.error("Configure OPENROUTER_MODEL in Streamlit Secrets.")
    st.stop()

# Import your existing agent functions
from agent import load_pages, extract, synthesize, report

uploaded_file = st.file_uploader(
    "Choose a document",
    type=["pdf", "txt"],
)

company = st.text_input(
    "Organization",
    value=os.getenv("COMPANY_CONTEXT", "a generic company"),
)

if st.button("Analyze Document", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a document first.")
        st.stop()

    temp_path = None

    try:
        suffix = Path(uploaded_file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix
        ) as temp:
            temp.write(uploaded_file.getvalue())
            temp_path = temp.name

        os.environ["COMPANY_CONTEXT"] = company

        with st.spinner("Analyzing your document..."):
            pages = load_pages(temp_path)
            evidence = extract(pages)
            verified_evidence, result = synthesize(evidence)

            report(
                verified_evidence,
                result,
                out="output/report.md",
            )

        st.subheader("Sentiment Verdict")
        st.write(result.get("verdict", "Not available"))

        st.subheader("Summary")
        st.write(result.get("summary", "Not available"))

        st.subheader("Verified Evidence")
        if verified_evidence:
            st.dataframe(
                verified_evidence,
                use_container_width=True,
            )
        else:
            st.info("No verified non-neutral evidence found.")

        st.subheader("Recommended Actions")
        actions = result.get("actions", [])
        if actions:
            st.dataframe(actions, use_container_width=True)
        else:
            st.info("No recommended actions returned.")

        report_path = Path("output/report.md")
        if report_path.exists():
            st.download_button(
                "Download Sentiment Report",
                data=report_path.read_text(encoding="utf-8"),
                file_name="sentiment_report.md",
                mime="text/markdown",
            )

    except Exception as e:
        st.error("Analysis failed.")
        st.code(str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
