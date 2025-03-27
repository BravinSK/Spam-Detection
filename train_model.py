import numpy as np
import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_and_preprocess_data

def train_spam_classifier():
    """
    Train a Naive Bayes classifier for spam detection
    
    Returns:
        tuple: Trained model, classification report, confusion matrix
    """
    # Load and preprocess data
    X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess_data('data/spam_dataset.csv')
    
    # Train Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Generate evaluation metrics
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save model and vectorizer
    joblib.dump(classifier, 'models/spam_classifier_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    return classifier, report, conf_matrix

# Run training
if __name__ == '__main__':
    model, report, conf_matrix = train_spam_classifier()
    print("Classification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(conf_matrix)