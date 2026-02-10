# Fake News Detector

A fake news detection system built with Python, featuring an SVM machine learning model , combined with NewsAPI and Google Fact Check Tools API. The system analyzes text, PDFs, and uploaded files to provide verdicts on news authenticity.

## Project Structure

```
Fake-news-detector/
├── backend/              # Backend logic (Flask or similar, if used)
├── data/                 # Datasets (Fake.csv, Real.csv) and trained models
├── frontend/             # Streamlit web app (app.py)
├── model/                # Model training scripts
├── preprocessing/        # Text cleaning and preprocessing utilities
├── utils/                # Additional utilities (e.g., text extraction)
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup
├── README.md             # This file
└── .env                  # Environment variables (API keys, not committed)
```
