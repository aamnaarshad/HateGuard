# ml/predictor.py
# --------------------------------------------------
# This file loads the saved (trained) models and
# uses them to make predictions on new text.
# --------------------------------------------------

import joblib
import os
from ml.preprocessor import clean_text

# ── Paths to the saved model files ──────────────────
MODELS_DIR = 'models'

VECTORIZER_PATH  = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
LR_MODEL_PATH    = os.path.join(MODELS_DIR, 'logistic_regression.pkl')
SVM_MODEL_PATH   = os.path.join(MODELS_DIR, 'svm.pkl')
RF_MODEL_PATH    = os.path.join(MODELS_DIR, 'random_forest.pkl')

# ── Label mapping ────────────────────────────────────
# Davidson dataset uses: 0 = Hate Speech, 1 = Offensive, 2 = Neither
LABEL_MAP = {
    0: 'Hate Speech',
    1: 'Offensive Language',
    2: 'Neither'
}

# ── Load everything once when the module is imported ─
# (Loading from disk every prediction would be slow)
vectorizer         = joblib.load(VECTORIZER_PATH)
logistic_model     = joblib.load(LR_MODEL_PATH)
svm_model          = joblib.load(SVM_MODEL_PATH)
random_forest_model = joblib.load(RF_MODEL_PATH)

# Map model names (used in the UI) to the actual model objects
MODELS = {
    'Logistic Regression': logistic_model,
    'Support Vector Machine': svm_model,
    'Random Forest': random_forest_model,
}


def predict(text):
    """
    Takes raw input text and returns predictions from all three models.

    Returns a dict like:
    {
        'cleaned_text': 'cleaned version of input',
        'results': [
            {'model': 'Logistic Regression', 'label': 'Offensive Language', 'confidence': 87.3},
            ...
        ]
    }
    """

    # Step 1: Clean the text the same way we cleaned training data
    cleaned = clean_text(text)

    # Step 2: Convert text to TF-IDF numbers the models understand
    # reshape(1, -1) means "treat this as ONE sample"
    features = vectorizer.transform([cleaned])

    results = []

    for model_name, model in MODELS.items():
        # Get the predicted class (0, 1, or 2)
        prediction = model.predict(features)[0]

        # Get probabilities for each class (returns array of 3 numbers)
        probabilities = model.predict_proba(features)[0]

        # The confidence is the probability of the predicted class
        confidence = round(probabilities[prediction] * 100, 1)

        results.append({
            'model':      model_name,
            'label':      LABEL_MAP[prediction],
            'confidence': confidence
        })

    return {
        'original_text': text,
        'cleaned_text':  cleaned,
        'results':       results
    }
