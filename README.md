# Spam Detection Project

This project is a machine learning-based spam detection system that classifies messages as either spam or ham (not spam). The project includes preprocessing, feature extraction, and a simple user interface for testing the model.

## Features

- Preprocessing of text data (cleaning, tokenization, and stopword removal).
- Feature extraction using TF-IDF vectorization.
- Classification of SMS messages as spam or ham.
- Simple and responsive user interface.

## Dataset

The dataset used for this project is the **SMS Spam Collection Dataset**, which can be downloaded from Kaggle:  
[Download Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)

## Project Structure

```
spam-detection-project/
│
├── preprocessing.py       # Python script for preprocessing and feature extraction
├── static/
│   └── styles.css         # CSS file for styling the user interface
├── templates/
│   └── index.html         # HTML file for the user interface
├── app.py                 # Flask application for running the project
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## How to Run the Project

1. Clone the repository:

   ```bash
   git clone https://github.com/BravinSK/Spam-Detection.git
   cd Spam-Detection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download) and place it in the `data/` directory.

4. Run the Flask application:

   ```bash
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000` to use the application.

## Technologies Used

- **Python**: For preprocessing and backend logic.
- **Flask**: For building the web application.
- **HTML/CSS**: For the user interface.
- **NLTK**: For text preprocessing.
- **Scikit-learn**: For feature extraction and model building.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset?resource=download)
- Libraries: NLTK, Scikit-learn, Flask
