import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from preprocessing import clean_text
from model import get_model

# Load dataset
df = pd.read_csv("../data/emails.csv")

# Preprocess
df['text'] = df['text'].apply(clean_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = get_model()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
