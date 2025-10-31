# ==============================================================
#  Fake News Detector API (Render-ready, Username-based)
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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

# ==============================================================
#  CONFIGURATION
# ==============================================================
load_dotenv()

# ✅ Ensure TextBlob corpora are available (auto-download)
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import textblob.download_corpora as download
    logging.info("⬇️ Downloading missing TextBlob corpora...")
    download.download_all()
    logging.info("✅ TextBlob corpora downloaded successfully.")

# ==============================================================
#  FLASK APP SETUP
# ==============================================================
app = Flask(__name__)
CORS(app, origins=["*"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "users.db"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model2.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl"))

# ==============================================================
#  LOGGING
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==============================================================
#  LOAD MODEL + VECTORIZER
# ==============================================================
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")
    model, vectorizer = None, None

# ==============================================================
#  DATABASE SETUP
# ==============================================================
def init_db():
    """Initialize SQLite user database."""
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.commit()

init_db()

def get_db_connection():
    return sqlite3.connect(app.config["DB_PATH"])

# ==============================================================
#  HELPERS
# ==============================================================
def tokenize_text(text: str) -> str:
    """Clean and normalize input text."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_fake_news(text: str) -> dict:
    """Predict fake vs real news using ML model."""
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
    """Analyze simple text heuristics."""
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

def verify_jwt(token: str):
    """Verify JWT token and return username if valid."""
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

# ==============================================================
#  ROUTES
# ==============================================================
@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API is running. Use /predict to scan text."})

# -------------------- AUTH ROUTES --------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    hashed_pw = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
        logging.info(f"🟢 Registered new user: {username}")
        return jsonify({"message": "User registered successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 400
    except Exception as e:
        logging.error(f"Database error: {e}")
        return jsonify({"error": "Internal error"}), 500

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username = ?", (username,))
        row = cur.fetchone()

    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid credentials"}), 401

    token = jwt.encode({
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=2)
    }, app.config["JWT_SECRET"], algorithm="HS256")

    logging.info(f"🔑 User logged in: {username}")
    return jsonify({"token": token, "username": username})

# -------------------- PREDICTION ROUTE --------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # JWT is optional (guest mode allowed)
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)

        data = request.get_json() or {}
        text = data.get("text", "")
        headline = data.get("headline", "")
        url = data.get("url", "")

        if not text:
            return jsonify({"error": "Missing text"}), 400

        ml_result = predict_fake_news(text)
        heuristics = analyze_text_heuristics(text)

        blob = TextBlob(text)
        sentences = []
        for sent in blob.sentences:
            sent_text = str(sent)
            sent_pred = predict_fake_news(sent_text)
            if "error" in sent_pred:
                continue
            sentences.append({
                "text": sent_text,
                "prediction": "Fake" if sent_pred["prediction"] == "0" else "Real",
                "confidence": sent_pred["confidence"]
            })

        response = {
            "username": username or "guest",
            "headline": headline,
            "url": url,
            "prediction": "Fake" if ml_result["prediction"] == "0" else "Real",
            "confidence": ml_result["confidence"],
            "class_probs": ml_result["class_probs"],
            "heuristics": heuristics,
            "sentences": sentences
        }

        logging.info(f"🧠 Scan complete by {username or 'guest'}: {headline or '[No Headline]'} ({response['prediction']})")
        return safe_json(response)

    except Exception as e:
        logging.exception("❌ Prediction error:")
        return jsonify({"error": str(e)}), 500

# -------------------- FULL REPORT --------------------
@app.route("/full-report")
def full_report():
    return app.send_static_file("full-report.html")

# ==============================================================
#  ERROR HANDLERS
# ==============================================================
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logging.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500

# ==============================================================
#  MAIN ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
