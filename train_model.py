# train_model.py
# --------------------------------------------------
# Run this script ONCE to train all three ML models
# and save them to the /models folder.
#
# Usage:  python train_model.py
# --------------------------------------------------

import os
import pandas as pd
import joblib
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model         import LogisticRegression
from sklearn.svm                  import LinearSVC
from sklearn.ensemble             import RandomForestClassifier
from sklearn.model_selection      import train_test_split
from sklearn.metrics              import classification_report
from sklearn.calibration          import CalibratedClassifierCV

from ml.preprocessor import clean_text

# ── 0. Make sure NLTK data is available ─────────────
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

# ── 1. Load the dataset ──────────────────────────────
# Download from: https://github.com/t-davidson/hate-speech-and-offensive-language
# File: labeled_data.csv   →  place it in the /data folder
DATA_PATH = os.path.join('data', 'labeled_data.csv')

print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# The column 'class' contains: 0=Hate Speech, 1=Offensive, 2=Neither
# The column 'tweet' contains the raw tweet text
print(f"   Total records: {len(df)}")
print(f"   Class distribution:\n{df['class'].value_counts()}\n")

# ── 2. Preprocess all tweets ─────────────────────────
print("🧹 Cleaning tweets (this may take a minute)...")
df['cleaned'] = df['tweet'].apply(clean_text)
print("   Done!\n")

# ── 3. Split into features (X) and labels (y) ────────
X = df['cleaned']   # text
y = df['class']     # 0, 1, or 2

# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,     # makes results reproducible
    stratify=y           # keep class proportions the same in both splits
)

# ── 4. TF-IDF Vectorization ──────────────────────────
# Converts text into numbers.
# ngram_range=(1,2) means use single words AND pairs of words.
# max_features limits vocabulary size to top 50,000 terms.
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True    # use log scaling — helps with imbalanced classes
)

X_train_tfidf = vectorizer.fit_transform(X_train)  # learn vocabulary + transform
X_test_tfidf  = vectorizer.transform(X_test)        # transform only (no learning)
print("   Done!\n")

# ── 5. Train Model 1: Logistic Regression ────────────
print("Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,    # allow enough iterations to converge
    C=1.0, 
    class_weight='balanced',           # regularization strength
    random_state=42
)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
print("   Logistic Regression Results:")
print(classification_report(y_test, lr_preds,
      target_names=['Hate Speech', 'Offensive', 'Neither']))

# ── 6. Train Model 2: SVM ────────────────────────────
# LinearSVC is fast but doesn't support predict_proba natively,
# so we wrap it with CalibratedClassifierCV to get probabilities.
print("🤖 Training Support Vector Machine...")
svm_base  = LinearSVC(C=1.0, max_iter=2000, class_weight='balanced',random_state=42)
svm_model = CalibratedClassifierCV(svm_base, cv=3)   # adds probability support
svm_model.fit(X_train_tfidf, y_train)
svm_preds = svm_model.predict(X_test_tfidf)
print("   SVM Results:")
print(classification_report(y_test, svm_preds,
      target_names=['Hate Speech', 'Offensive', 'Neither']))

# ── 7. Train Model 3: Random Forest ──────────────────
print("🤖 Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,    # number of decision trees
    max_depth=None,      # trees can grow as deep as needed
    n_jobs=-1,           # use all CPU cores
    class_weight='balanced',
    random_state=42,
)
rf_model.fit(X_train_tfidf, y_train)
rf_preds = rf_model.predict(X_test_tfidf)
print("   Random Forest Results:")
print(classification_report(y_test, rf_preds,
      target_names=['Hate Speech', 'Offensive', 'Neither']))

# ── 8. Save everything to /models ────────────────────
os.makedirs('models', exist_ok=True)

print("\n💾 Saving models to /models folder...")
joblib.dump(vectorizer,  'models/tfidf_vectorizer.pkl')
joblib.dump(lr_model,    'models/logistic_regression.pkl')
joblib.dump(svm_model,   'models/svm.pkl')
joblib.dump(rf_model,    'models/random_forest.pkl')

print("\n✅ All done! You can now run:  python app.py")
