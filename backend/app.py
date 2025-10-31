# app.py
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)  # Suppress TextBlob warnings

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib
import re
import os
import sqlite3
import logging
import json
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv

# -------------------- CONFIG --------------------
load_dotenv()  # Load .env if available

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", "users.db")

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model2.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl"))

# -------------------- LOGGING --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# -------------------- LOAD ML MODEL --------------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")
    model, vectorizer = None, None

# -------------------- DATABASE --------------------
def init_db() -> None:
    """Initialize the SQLite database and create tables if they don't exist."""
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(app.config["DB_PATH"])

# -------------------- HELPERS --------------------
def tokenize_text(text: str) -> str:
    """Clean and normalize input text."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower()

def predict_fake_news(text: str) -> dict:
    """Predict whether the text is fake news using the ML model."""
    if not model or not vectorizer:
        return {"error": "Model not loaded"}

    processed = tokenize_text(text)
    features = vectorizer.transform([processed])
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    confidence = float(max(probs))

    return {
        "prediction": str(prediction),
        "confidence": confidence,
        "class_probs": {"0": float(probs[0]), "1": float(probs[1])}
    }

def analyze_text_heuristics(text: str) -> dict:
    """Analyze text heuristics to compute a fake-score."""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    word_count = len(words)
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, word_count)
    exclamations = text.count("!")

    fake_score = (
        0.4 * (1 - abs(sentiment)) +
        0.3 * uppercase_ratio +
        0.3 * min(exclamations / 5, 1)
    )
    return {
        "sentiment": sentiment,
        "uppercase_ratio": uppercase_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score
    }

def safe_json(data: dict):
    """Ensure safe JSON serialization."""
    return jsonify(json.loads(json.dumps(data, default=str)))

def verify_jwt(token: str) -> str | None:
    """Verify JWT token and return email if valid."""
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("email")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# -------------------- HOME ROUTE --------------------
@app.route("/")
def home():
    return "Fake News Detector API is running. Use /predict to scan text."

# -------------------- AUTH ROUTES --------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    hashed_pw = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (email, hashed_pw))
            conn.commit()
        logging.info(f"🟢 Registered new user: {email}")
        return jsonify({"message": "User registered successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered."}), 400
    except Exception as e:
        logging.error(e)
        return jsonify({"error": "Internal error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    email = data.get("email", "").strip()
    password = data.get("password", "").strip()

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE email = ?", (email,))
        row = cur.fetchone()

    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({
        "email": email,
        "exp": datetime.utcnow() + timedelta(hours=2)
    }, app.config["JWT_SECRET"], algorithm="HS256")

    logging.info(f"🔑 User logged in: {email}")
    return jsonify({"token": token, "email": email})

# -------------------- SCAN ROUTE --------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json() or {}
        text = data.get("text", "")
        headline = data.get("headline", "")
        url = data.get("url", "")

        if not text:
            return jsonify({"error": "Missing text"}), 400

        ml_result = predict_fake_news(text)
        heuristics = analyze_text_heuristics(text)

        if "error" in ml_result:
            return jsonify(ml_result), 500

        response = {
            "headline": headline,
            "url": url,
            "prediction": "Fake" if ml_result["prediction"] == "0" else "Real",
            "confidence": ml_result["confidence"],
            "class_probs": ml_result["class_probs"],
            "heuristics": heuristics
        }
        logging.info(f"🧠 Scan complete: {headline or '[No Headline]'} ({response['prediction']})")
        return safe_json(response)

    except Exception as e:
        logging.exception("❌ Prediction error:")
        return jsonify({"error": str(e)}), 500

# -------------------- FULL REPORT --------------------
@app.route("/full-report")
def full_report():
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Full Report - Fake News Detector</title>
      <style>
        body { font-family: Arial; margin: 40px; background:#f7f9fc; color:#333; }
        .container { max-width: 800px; margin:auto; background:white; padding:30px; border-radius:10px; box-shadow:0 4px 12px rgba(0,0,0,0.1);}
        h2 { color:#1e5bc7; }
        table { width:100%; border-collapse:collapse; margin-top:20px; }
        td,th { border:1px solid #ccc; padding:10px; text-align:left; }
        th { background:#f0f4ff; }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>Fake News Detector - Full Report</h2>
        <p>View your recent scans and their predictions below.</p>
        <p><i>Data stored locally in your browser’s extension.</i></p>
      </div>
    </body>
    </html>
    """
    return render_template_string(template)

# -------------------- ERROR HANDLER --------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logging.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500

# -------------------- MAIN --------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
