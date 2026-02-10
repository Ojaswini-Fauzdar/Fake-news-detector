import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def preprocess(text):
    """
    Clean and preprocess the text:
    - remove special characters
    - convert to lowercase
    - remove extra spaces
    - remove stopwords
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = re.sub(r'\W', ' ', text)          
    text = re.sub(r'\s+', ' ', text)         
    text = text.lower().strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


# Preparing data
DATA_FOLDER = 'data'

fake_path = os.path.join(DATA_FOLDER, 'Fake.csv')
true_path = os.path.join(DATA_FOLDER, 'True.csv')


if not os.path.exists(fake_path):
    raise FileNotFoundError(f"File not found: {fake_path}")
if not os.path.exists(true_path):
    raise FileNotFoundError(f"File not found: {true_path}")

print("Loading datasets...")
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

#1 = Fake, 0 = Real
fake['label'] = 1
true['label'] = 0


data = pd.concat([fake, true], ignore_index=True)


print("Combining title and text...")
data['full_text'] = data['title'].fillna('') + ' ' + data['text'].fillna('')

# Preprocess text
print("Preprocessing text...")
data['processed_text'] = data['full_text'].apply(preprocess)


data = data[data['processed_text'].str.strip() != ''].reset_index(drop=True)

print(f"Total samples after cleaning: {len(data)}")




print("Creating TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=8000,          
    ngram_range=(1, 2),         
    min_df=5,                   
    max_df=0.95                 
)

X = vectorizer.fit_transform(data['processed_text'])
y = data['label']

print(f"Vectorized shape: {X.shape}")




print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


#train model
print("Training LinearSVC...")
model = LinearSVC(
    max_iter=2000,
    C=1.0,
    random_state=42
)

model.fit(X_train, y_train)
#evaluating
print("Evaluating model...")
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=['Real', 'Fake']))


print("\nSaving model and vectorizer...")
joblib.dump(model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Done!")
print("Files created:")
print("  - svm_model.pkl")
print("  - vectorizer.pkl")
