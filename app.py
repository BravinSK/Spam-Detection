from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
from preprocessing import preprocess_text

app = Flask(__name__)

# Load pre-trained model and vectorizer
model = joblib.load('models/spam_classifier_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

@app.route('/', methods=['GET'])
def home():
    """
    Render the home page
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    """
    Predict if an email is spam
    
    Returns:
        JSON response with prediction result
    """
    try:
        # Get email text from request
        email_text = request.form['email_text']
        
        # Preprocess the text
        cleaned_text = preprocess_text(email_text)
        
        # Vectorize the text
        vectorized_text = vectorizer.transform([cleaned_text]).toarray()
        
        # Predict
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        
        # Prepare response
        result = {
            'is_spam': bool(prediction),
            'spam_probability': float(max(probability)),
            'message': 'Spam detected!' if prediction else 'Not spam.'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)