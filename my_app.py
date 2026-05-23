# app.py
from flask import Flask, render_template, request, jsonify
from ml.predictor import predict

app = Flask(__name__)

MIN_WORDS = 3   # must match the value in script.js

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'Please enter some text.'}), 400

    # Minimum word count check
    word_count = len(text.split())
    if word_count < MIN_WORDS:
        return jsonify({
            'error': f'Please enter at least {MIN_WORDS} words. '
                     f'A single word has no context for the model to classify reliably.'
        }), 400

    result = predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)