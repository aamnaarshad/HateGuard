# app.py
# --------------------------------------------------
# The main Flask web application.
# Run with:  python app.py
# Then open: http://127.0.0.1:5000
# --------------------------------------------------

from flask import Flask, render_template, request, jsonify
from ml.predictor import predict

# Create the Flask app
app = Flask(__name__)


# ── Route 1: Home page ───────────────────────────────
@app.route('/')
def index():
    """Render the main page with the input form."""
    return render_template('index.html')


# ── Route 2: Prediction API ──────────────────────────
@app.route('/predict', methods=['POST'])
def predict_route():
    """
    Receives text from the form via AJAX (POST request),
    runs it through all three models, and returns JSON results.
    """

    # Get the text the user typed
    data = request.get_json()
    text = data.get('text', '').strip()

    # Basic validation — don't process empty text
    if not text:
        return jsonify({'error': 'Please enter some text.'}), 400

    # Run prediction
    result = predict(text)

    # Return results as JSON to the frontend
    return jsonify(result)


# ── Start the server ─────────────────────────────────
if __name__ == '__main__':
    # debug=True means the server auto-restarts when you edit code
    # Set debug=False when deploying to production
    app.run(debug=True)
