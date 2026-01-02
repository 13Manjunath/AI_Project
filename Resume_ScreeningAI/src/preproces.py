import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # Convert multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)

    # Lowercase
    text = text.lower()

    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

