# Fake News Detector

A fake news detection system that verifies real-world claims, news articles, or social media posts that ingests claims (user-submitted text, article URLs, or social media posts), retrieves evidence from multiple live and trusted sources (fact-check APIs, news outlets), and produces a verdict (True / False / Misleading / Unverified) with a confidence score.


## Getting started

1. Clone the repository:
```bash
git clone <repository-url>
cd fake-news-detector
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Web Application

```bash
python web_app/app.py
```

