import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download the NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess email text for spam detection
    
    Args:
        text (str): Input email text
    
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into a string
    return ' '.join(tokens)

def load_and_preprocess_data(filepath):
    """
    Load and preprocess spam dataset
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        tuple: Preprocessed features and labels
    """
    # Load dataset with an alternate encoding
    try:
        df = pd.read_csv(filepath, encoding='utf-8')  # Try utf-8 first
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin1')  # Fallback to latin1
    
    
    df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
    
    # Convert labels to binary (e.g., 'spam' -> 1, 'ham' -> 0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})
    
    # Preprocess text
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_text']).toarray()
    y = df['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, vectorizer

# Example usage
if __name__ == '__main__':
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data('data/spam_dataset.csv')