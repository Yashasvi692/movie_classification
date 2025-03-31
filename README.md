# Movie Genre Classification

## Overview
This project predicts movie genres using a Logistic Regression model tuned with Grid Search, based on movie descriptions.

## Dataset
- **Training Data**: `train_data.txt` (54,214 samples, 27 genres).
- **Test Data**: `test_data.txt` (54,200 samples).

## Methodology
- **Preprocessing**: Lowercased, removed punctuation/numbers, lemmatized, removed stop words.
- **Features**: TF-IDF with 10,000 features from descriptions.
- **Model**: Logistic Regression (`C=10.0`, `class_weight='balanced'`), trained on 43,371 samples, validated on 10,843.
- **Output**: `submission.csv` with predictions.

## Results
- **Validation Accuracy**: 53.95%.
- **Cross-Validation Accuracy**: 53.75%.

## Deployment
- **API**: Flask app running locally at `http://127.0.0.1:5000/predict`.
- **Endpoint**: POST `/predict`, e.g., `{"description": "A hacker discovers a simulated reality."}`.
- **Response**: `{"genre": "sci-fi", "confidence": 0.XX}`.

## Testing
- **Tool**: Postman
- **Request**:
  - Method: POST
  - URL: `http://127.0.0.1:5000/predict`
  - Headers: `Content-Type: application/json`
  - Body: `{"description": "A hacker discovers a simulated reality."}`
- **Response**: `{"genre": "sci-fi", "confidence": 0.85}`

## Usage
- Run `app.py` locally with Python.
- Dependencies: `numpy`, `pandas`, `nltk`, `sklearn`, `flask`, `joblib`.

## Files
- `submission.csv`: Test predictions.
- `movie_genre_model.pkl`: Saved model.
- `tfidf_vectorizer.pkl`: Saved vectorizer.
- `label_encoder.pkl`: Saved label encoder.
