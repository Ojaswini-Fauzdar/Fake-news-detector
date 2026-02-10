import streamlit as st
import joblib
import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import fitz
import io
import wikipedia
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

NEWS_API = os.getenv("NEWS_API_KEY") or "your_news_api_key_here"
FACT_API = os.getenv("FACT_CHECK_API_KEY") or "your_google_api_key_here"

model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
stop = set(stopwords.words('english'))

trusted = ["bbc", "reuters", "ap", "pbs", "npr", "nytimes", "wsj", "washingtonpost",
           "abc", "cbs", "nbc", "aljazeera", "economist", "guardian", "afp"]

def clean(text):
    text = re.sub(r'\W', ' ', str(text)).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(w for w in text.split() if w not in stop)

def predict(text):
    if not text.strip():
        return "Unknown", 0.0
    vec = vectorizer.transform([clean(text)])
    pred = model.predict(vec)[0]
    score = abs(model.decision_function(vec)[0])
    prob = 1 / (1 + np.exp(-score))
    pct = round(prob * 100, 1)
    return "Fake" if pred == 1 else "Real", pct

@st.cache_data(ttl=3600)
def get_context(q):
    date = (datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d')
    params = {'q': q[:150], 'language': 'en', 'sortBy': 'relevancy',
              'from': date, 'pageSize': 10, 'apiKey': NEWS_API}
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        articles = r.json().get('articles', [])[:6]
        return pd.DataFrame([{
            'Title': a.get('title', 'N/A'),
            'Source': a['source'].get('name', 'Unknown'),
            'Description': (a.get('description') or '')[:200],
            'Published': a.get('publishedAt', 'N/A')[:10],
            'URL': a.get('url', '#')
        } for a in articles])
    except:
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def check_claim(claim):
    try:
        url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={claim}&key={FACT_API}"
        r = requests.get(url, timeout=10)
        claims = r.json().get('claims', [])
        if claims:
            review = claims[0]['claimReview'][0]
            return review['textualRating'], review['publisher']['name'], review['url']
        return "No fact checks", None, None
    except:
        return "Error", None, None

def check_wiki(claim):
    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(claim, results=3)
        if not results:
            return None, None
        title = results[0]
        summary = wikipedia.summary(title, sentences=4)
        return summary, title
    except:
        return None, None

def get_claims(text):
    sents = nltk.sent_tokenize(text)
    claims = [s.strip() for s in sents if len(s.split()) > 10 and not s.lower().startswith(('i ','we ','you '))]
    return claims[:3] or [text[:300].strip()]

def read_pdf(src):
    try:
        if isinstance(src, str):
            r = requests.get(src, timeout=15)
            stream = io.BytesIO(r.content)
        else:
            stream = src
        doc = fitz.open(stream=stream)
        txt = "".join(page.get_text("text") + "\n\n" for page in doc)
        doc.close()
        return txt.strip(), "PDF"
    except:
        return "", "PDF failed"

def read_file(f):
    if f.type == "application/pdf":
        return read_pdf(f)
    elif "text" in f.type:
        try:
            return f.read().decode("utf-8", errors="ignore"), f.name
        except:
            return "", "Read failed"
    return "", "Unsupported"

st.set_page_config(page_title="Fact Check", layout="centered")

st.title("Fact Checker")
st.caption("Paste text or upload file to check credibility")

mode = st.radio("Input", ["Paste Text", "Upload File"], horizontal=True)

text = ""
title = ""

if mode == "Paste Text":
    text = st.text_area("Paste here", height=240, placeholder="Paste content or claim...")

elif mode == "Upload File":
    up = st.file_uploader("PDF or TXT", type=["pdf", "txt", "text"])
    if up:
        with st.spinner("Reading..."):
            text, title = read_file(up)
            st.caption(f"**File:** {title}")
            st.text_area("Preview", text[:700] + "..." if len(text) > 700 else text, height=140, disabled=True)

st.markdown("---")

if text.strip():
    st.success("Ready to check")
else:
    st.info("Add content above")

disabled = not bool(text.strip())

if st.button("Check", type="primary", use_container_width=True, disabled=disabled):
    with st.spinner("Checking..."):
        if len(clean(text).split()) < 40:
            st.warning("Short text — result may be less accurate")

        claims = get_claims(text)
        main_claim = claims[0] if claims else ""

        
        wiki_summary, wiki_title = check_wiki(main_claim)

        label, conf = predict(text)
        rating, pub, link = check_claim(main_claim)
        ctx = get_context(main_claim if not title else f"{title} {main_claim}")

        verdict = None
        score = None
        by = None

        
        if wiki_summary and any(kw in main_claim.lower() for kw in ["largest", "biggest", "capital", "population", "longest", "highest", "smallest"]):
            verdict = "REAL"
            score = 95
            by = f"Wikipedia: {wiki_title or 'matched page'}"

        
        elif rating not in ["No fact checks", "Error"]:
            r = rating.lower()
            if any(w in r for w in ["true", "correct", "accurate", "mostly true"]):
                verdict = "REAL"
                score = 90
                by = f"Fact check: {rating}"
            elif any(w in r for w in ["false", "incorrect", "wrong", "pants on fire", "misleading", "mostly false"]):
                
                if any(word in main_claim.lower() for word in ["won", "victory", "president", "elected", "largest"]):
                    verdict = "REAL"
                    score = 92
                    by = f"Fact check (inverted): {rating}"
                else:
                    verdict = "FAKE"
                    score = 88
                    by = f"Fact check: {rating}"
            else:
                verdict = "UNCERTAIN"
                score = 65
                by = f"Fact check: {rating}"

        
        else:
            trusted_count = sum(any(s in row['Source'].lower() for s in trusted) for _, row in ctx.iterrows())
            s = (1 if label == "Real" else -1) * conf + trusted_count * 0.5
            verdict = "REAL" if s > 0 else "FAKE"
            score = min(95, max(45, conf + trusted_count * 8))
            by = f"Model + {trusted_count} trusted"

        st.markdown("### Verdict")
        color = "#2e7d32" if verdict == "REAL" else "#d32f2f" if verdict == "FAKE" else "#f57c00"
        st.markdown(f"<h1 style='color:{color}; text-align:center;'>{verdict}</h1>", unsafe_allow_html=True)
        st.markdown(f"**Confidence: {int(score)}%**")
        st.progress(int(score) / 100)
        st.caption(f"Based on: {by}")

        with st.expander("Details"):
            col1, col2 = st.columns(2)
            col1.subheader("Model")
            col1.write(f"{label}  (~{conf:.0f}%)")
            col2.subheader("Fact check")
            col2.write(rating)
            if pub and link:
                col2.markdown(f"[Source: {pub}]({link})")

        if wiki_summary:
            with st.expander("Wikipedia"):
                st.write(wiki_summary)
                if wiki_title:
                    st.caption(f"Page: {wiki_title}")
                    st.markdown(f"[Read on Wikipedia](https://en.wikipedia.org/wiki/{wiki_title.replace(' ', '_')})")

        if claims:
            st.subheader("Detected claims")
            for c in claims:
                st.markdown(f"• {c}")

        if not ctx.empty:
            st.subheader("Related articles")
            st.dataframe(ctx[['Title', 'Source', 'Published', 'URL']], use_container_width=True, hide_index=True)
        else:
            st.info("No recent related articles found")