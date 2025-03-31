from flask import Flask, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

model = joblib.load('movie_genre_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

@app.route('/')
def index():
    return "Welcome to the Movie Genre Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'description' not in data:
            return jsonify({'error': 'Missing description'}), 400
        description = data['description']
        cleaned_text = clean_text(description)
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction_encoded = model.predict(text_vectorized)
        # Strip spaces from prediction
        cleaned_prediction = [pred.strip() for pred in prediction_encoded]
        print(f"Raw prediction: {prediction_encoded}, Cleaned: {cleaned_prediction}")
        prediction = le.inverse_transform(cleaned_prediction)[0]
        probability = model.predict_proba(text_vectorized).max()
        return jsonify({'genre': prediction, 'confidence': float(probability)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)