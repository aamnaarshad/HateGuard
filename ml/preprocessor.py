# ml/preprocessor.py
# --------------------------------------------------
# This file handles ALL text cleaning before we
# pass any text to the machine learning models.
# --------------------------------------------------

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only runs once)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Create a lemmatizer — converts words to their base form
# e.g. "running" -> "run", "better" -> "good"
lemmatizer = WordNetLemmatizer()

# Load English stopwords ("the", "is", "at", etc.)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Takes a raw tweet/sentence and returns a cleaned version.

    Steps:
    1. Lowercase everything
    2. Remove URLs
    3. Remove Twitter handles (@username)
    4. Remove hashtag symbols (keep the word)
    5. Remove punctuation and numbers
    6. Tokenize (split into words)
    7. Remove stopwords
    8. Lemmatize each word
    9. Rejoin into a single string
    """

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs (http://... or https://...)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: Remove Twitter handles like @username
    text = re.sub(r'@\w+', '', text)

    # Step 4: Remove the '#' symbol but keep the word
    text = re.sub(r'#', '', text)

    # Step 5: Remove anything that is NOT a letter or space
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 6: Split into individual words
    words = text.split()

    # Step 7 & 8: Remove stopwords AND lemmatize in one loop
    cleaned_words = []
    for word in words:
        if word not in stop_words:           # skip common words
            lemma = lemmatizer.lemmatize(word)  # get base form
            cleaned_words.append(lemma)

    # Step 9: Rejoin words into a single string
    return ' '.join(cleaned_words)
