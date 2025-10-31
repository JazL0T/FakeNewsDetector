# ==============================================================
# Fake News Detector API (Final Secure Version - Render Ready)
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template_string, send_from_directory
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
import tldextract

# ============================================================== #
# CONFIGURATION
# ============================================================== #
load_dotenv()
app = Flask(__name__)
CORS(app, origins=["*"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "users.db"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model2.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl"))

# ============================================================== #
# LOGGING
# ============================================================== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ============================================================== #
# LOAD MODEL
# ============================================================== #
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")
    model, vectorizer = None, None

# ============================================================== #
# DATABASE
# ============================================================== #
def init_db():
    """Initialize the SQLite user database."""
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

# ============================================================== #
# HELPERS
# ============================================================== #
def tokenize_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_fake_news(text: str) -> dict:
    """Predict fake vs real news using the model."""
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
    """Compute heuristic features."""
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

def compute_trustability(url: str) -> dict:
    """Basic website trustability estimation."""
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted_sources = ["bbc.com", "reuters.com", "cnn.com", "nytimes.com", "theguardian.com"]
    suspicious_sources = ["clickbait", "rumor", "gossip", "unknownblog"]

    trust_score = 50
    category = "Uncertain"

    if any(src in domain for src in trusted_sources):
        trust_score = 90
        category = "Trusted"
    elif any(src in domain for src in suspicious_sources):
        trust_score = 30
        category = "Suspicious"

    return {"domain": domain, "trust_score": trust_score, "category": category}

def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def safe_json(data: dict):
    return jsonify(json.loads(json.dumps(data, default=str)))

# ============================================================== #
# ROUTES
# ============================================================== #
@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API running ✅"})

# ----- AUTH -----
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
        return jsonify({"message": "User registered successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 400

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

    return jsonify({"token": token, "username": username})

# ----- PREDICTION -----
@app.route("/predict", methods=["POST"])
def predict():
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
    trust = compute_trustability(url)

    response = {
        "username": username or "Guest",
        "headline": headline,
        "url": url,
        "prediction": "Fake" if ml_result["prediction"] == "0" else "Real",
        "confidence": ml_result["confidence"],
        "class_probs": ml_result["class_probs"],
        "heuristics": heuristics,
        "trustability": trust
    }

    return safe_json(response)

# ----- FULL REPORT -----
@app.route("/full-report")
def full_report():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)

    if not username:
        # Not logged in — show friendly message
        html = """
        <html><head><title>Login Required</title></head>
        <body style='font-family:Arial;background:#f5f8ff;text-align:center;margin-top:100px;'>
          <div style='background:white;padding:40px;max-width:400px;margin:auto;border-radius:10px;box-shadow:0 4px 10px rgba(0,0,0,0.1);'>
            <h2 style='color:#1565c0;'>🔒 Login Required</h2>
            <p>You must be logged in to view your full report.</p>
            <a href="#" onclick="window.close()" style='background:#1976d2;color:white;padding:8px 12px;border-radius:8px;text-decoration:none;'>Close</a>
          </div>
        </body></html>
        """
        return html, 401

    return send_from_directory(os.path.join(os.path.dirname(__file__), "templates"), "full-report.html")

# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
