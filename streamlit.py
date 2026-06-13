import streamlit as st
from main import text_url, text_pdf, get_claims, search_claims, get_verdict

st.set_page_config(page_title="Fake News Detector", layout="centered")
from dotenv import load_dotenv

load_dotenv(override=True)

st.markdown("""
    <style>
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 780px; }
        h1 { text-align: center; font-size: 2.4rem !important; margin-bottom: 2.5rem !important; }
        .input-label { font-size: 1.15rem; font-weight: 600; margin-bottom: 0.3rem; }
        div[data-testid="stSelectbox"] { margin-bottom: 1.5rem; }
        div[data-testid="stTextInput"] > div > input,
        div[data-testid="stTextArea"] > div > textarea {
            font-size: 15px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Fake News Detector")


if "fetched_text" not in st.session_state:
    st.session_state.fetched_text = ""
if "last_url" not in st.session_state:
    st.session_state.last_url = ""

st.markdown('<p class="input-label">Choose input type</p>', unsafe_allow_html=True)
mode = st.selectbox("", ["Paste Text", "Enter URL", "Upload PDF"], label_visibility="collapsed")

st.markdown("---")

text = ""

if mode == "Paste Text":
    st.markdown('<p class="input-label">Article or claim</p>', unsafe_allow_html=True)
    text = st.text_area("", placeholder="Paste your article or claim here…", height=280, label_visibility="collapsed")
    st.session_state.fetched_text = ""
    st.session_state.last_url = ""

elif mode == "Enter URL":
    st.markdown('<p class="input-label">Article URL</p>', unsafe_allow_html=True)
    url = st.text_input("", placeholder="https://www.example.com/article", label_visibility="collapsed")

    if url and url != st.session_state.last_url:
        with st.spinner("Fetching article…"):
            fetched = text_url(url)
        if fetched:
            st.session_state.fetched_text = fetched
            st.session_state.last_url = url
        else:
            st.error("Failed to fetch article. Try another URL.")

    if st.session_state.fetched_text:
        with st.expander("Preview fetched content"):
            preview = st.session_state.fetched_text
            st.write(preview[:1500] + "…" if len(preview) > 1500 else preview)

    text = st.session_state.fetched_text

elif mode == "Upload PDF":
    st.markdown('<p class="input-label">Upload PDF</p>', unsafe_allow_html=True)
    file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
    if file:
        with st.spinner("Reading PDF…"):
            st.session_state.fetched_text = text_pdf(file)
    if st.session_state.fetched_text:
        st.success("PDF loaded successfully")
    text = st.session_state.fetched_text

st.markdown("&nbsp;", unsafe_allow_html=True)

if st.button(" Analyze", type="primary", use_container_width=True):
    if not text or not text.strip():
        st.error("Please provide text, a URL, or a PDF.")
    else:
        try:
            bar = st.progress(0, text="Extracting claims…")
            claims = get_claims(text)
            bar.progress(33, text=f"Searching web for {len(claims)} claims…")
            docs = search_claims(claims)
            bar.progress(66, text="Generating verdict…")
            result = get_verdict(text, docs)
            bar.progress(100, text="Done!")
            bar.empty()

            st.markdown("---")
            col1, col2 = st.columns([1, 2], gap="large")

            with col1:
                if result.verdict == "REAL":
                    st.success(f"**{result.verdict}**")
                elif result.verdict == "FAKE":
                    st.error(f"**{result.verdict}**")
                else:
                    st.warning(f"**{result.verdict}**")
                st.metric("Confidence", f"{result.confidence:.0%}")

            with col2:
                st.subheader("Explanation")
                st.write(result.explanation)

            if result.supporting_sources:
                with st.expander("Sources"):
                    for src in result.supporting_sources:
                        st.write(f"- {src}")

        except Exception as e:
            st.error("Analysis failed.")
            st.code(str(e))
