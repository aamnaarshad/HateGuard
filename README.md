<div align="center">

# ⬡ HateGuard

### Automated Hate Speech & Offensive Language Detection

A machine learning web app that classifies text into **Hate Speech**, **Offensive Language**, or **Neither** using three ML models — with confidence scores and majority verdict.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.8-154F5B?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-24%2C783%20tweets-6366f1?style=flat-square)

</div>

---

## Overview

HateGuard is a full-stack NLP application built as a Semester Project for the Machine Learning course. It tackles the **"Nuance Problem"** in content moderation — distinguishing targeted hate speech from casual profanity and genuinely neutral text.

The app trains three classical ML models on labeled tweet data, then serves predictions through a clean Flask web interface. All three models run simultaneously and a majority vote produces the final verdict.

---

## Features

- Three ML models running in parallel — Logistic Regression, SVM, Random Forest
- Confidence score with animated progress bar for each model
- Majority vote verdict across all three models
- Real-time text preprocessing preview (what the model actually sees)
- Example prompts to test different categories
- Balanced class weighting to reduce false positives on neutral text
- Clean, minimal web UI — no page reloads (AJAX)

---

## How It Works

```
User Input
    │
    ▼
Preprocessing        lowercase → strip URLs/@handles → remove stopwords → lemmatize
    │
    ▼
TF-IDF Vectorizer    converts cleaned text to numerical feature vector (50k features, bigrams)
    │
    ├──▶ Logistic Regression  ──┐
    ├──▶ Support Vector Machine ├──▶ Majority Vote ──▶ Final Verdict
    └──▶ Random Forest         ──┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Models | scikit-learn (LogisticRegression, LinearSVC, RandomForestClassifier) |
| NLP | NLTK (stopwords, lemmatizer), TF-IDF vectorizer |
| Frontend | HTML, CSS, Vanilla JavaScript (fetch API) |
| Model Persistence | joblib (.pkl files) |

---

## Model Performance

Evaluated on 20% held-out test split. `class_weight='balanced'` applied to all models to address the dataset's class imbalance (~77% offensive, ~17% neither, ~5% hate speech).

| Model | Hate Speech F1 | Offensive F1 | Neither F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | ~0.63 | ~0.94 | ~0.87 | ~91% |
| SVM | ~0.65 | ~0.95 | ~0.89 | ~92% |
| Random Forest | ~0.58 | ~0.93 | ~0.86 | ~90% |

> F1-Score is used over accuracy due to class imbalance. Hate Speech F1 is lower because it is the minority class in the dataset.

---

## Dataset

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.

Available on Kaggle: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

