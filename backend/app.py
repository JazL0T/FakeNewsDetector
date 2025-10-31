# ==============================================================
# Fake News Detector API (Final Secure Version - Render Ready)
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, send_from_directory
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

# ==============================================================
# CONFIGURATION
# ==============================================================
load_dotenv()
app = Flask(__name__)
CORS(app, origins=["*"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "users.db"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model2.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl"))

# ==============================================================
# LOGGING
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ==============================================================
# LOAD MODEL + VECTORIZER
# ==============================================================
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")
    model, vectorizer = None, None

# ==============================================================
# DATABASE INITIALIZATION
# ==============================================================
def init_db():
    """Ensure users.db exists and has the correct schema."""
    db_path = app.config["DB_PATH"]
    recreate = False

    if os.path.exists(db_path):
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                cur.execute("PRAGMA table_info(users)")
                cols = [c[1] for c in cur.fetchall()]
                if "username" not in cols or "password" not in cols:
                    logging.warning("⚠️ Outdated or invalid DB schema detected. Recreating users.db...")
                    recreate = True
        except Exception as e:
            logging.error(f"DB check failed ({e}), recreating...")
            recreate = True
    else:
        recreate = True

    if recreate:
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL
                    )
                """)
                conn.commit()
            logging.info("✅ users.db created successfully with correct schema.")
        except Exception as e:
            logging.error(f"❌ Failed to recreate users.db: {e}")

def get_db_connection():
    return sqlite3.connect(app.config["DB_PATH"])

# Initialize DB on startup
init_db()
try:
    with sqlite3.connect(app.config["DB_PATH"]) as _c:
        _cur = _c.cursor()
        _cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [t[0] for t in _cur.fetchall()]
        logging.info(f"🗄️ Existing tables: {tables}")
except Exception as _e:
    logging.error(f"Failed to list tables: {_e}")

# ==============================================================
# HELPERS
# ==============================================================
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_fake_news(text: str) -> dict:
    if not model or not vectorizer:
        return {"error": "Model not loaded"}
    features = vectorizer.transform([tokenize_text(text)])
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return {
        "prediction": str(prediction),
        "confidence": float(max(probs)),
        "class_probs": {"0": float(probs[0]), "1": float(probs[1])}
    }

def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = (0.4 * (1 - abs(sentiment)) + 0.3 * uppercase_ratio + 0.3 * min(exclamations / 5, 1))
    return {
        "sentiment": sentiment,
        "uppercase_ratio": uppercase_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score
    }

def compute_trustability(url: str) -> dict:
    try:
        domain = tldextract.extract(url or "").registered_domain or "unknown"
    except Exception:
        domain = "unknown"

    trusted_sources = ["bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com", "npr.org"]
    suspicious_markers = ["clickbait", "rumor", "gossip", "unknownblog", ".info", ".buzz", ".click"]

    score = 50
    if any(src in domain for src in trusted_sources):
        score = 90
        category = "Trusted"
    elif any(m in domain for m in suspicious_markers):
        score = 30
        category = "Suspicious"
    else:
        category = "Uncertain"

    return {"domain": domain, "trust_score": score, "category": category}

def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def safe_json(data: dict):
    return jsonify(json.loads(json.dumps(data, default=str)))

# ==============================================================
# ROUTES
# ==============================================================

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API running ✅"})

# ------------------ AUTH ------------------
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json(force=True) or {}
        username = data.get("username", "").strip()
        password = data.get("password", "").strip()

        if not username or not password:
            return jsonify({"error": "Missing username or password"}), 400

        hashed_pw = generate_password_hash(password)
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw))
            conn.commit()
        logging.info(f"🟢 Registered new user: {username}")
        return jsonify({"message": "User registered successfully."})

    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 400
    except Exception as e:
        logging.exception("❌ Registration error:")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json(force=True) or {}
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

    except Exception as e:
        logging.exception("❌ Login error:")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

# ------------------ PREDICTION ------------------
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

    ml = predict_fake_news(text)
    if "error" in ml:
        return jsonify(ml), 500

    heur = analyze_text_heuristics(text)
    trust = compute_trustability(url)

    return safe_json({
        "username": username or "Guest",
        "headline": headline,
        "url": url,
        "prediction": "Fake" if ml["prediction"] == "0" else "Real",
        "confidence": ml["confidence"],
        "class_probs": ml["class_probs"],
        "heuristics": heur,
        "trustability": trust
    })

# ------------------ FULL REPORT ------------------
@app.route("/full-report")
def full_report():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not token:
        token = request.args.get("token", "")  # ✅ allow token in URL query

    username = verify_jwt(token)
    if not username:
        html = """
        <!DOCTYPE html><html><head><meta charset="utf-8">
        <title>Login Required</title></head>
        <body style="font-family:Arial;background:#f5f8ff;color:#333;display:flex;justify-content:center;align-items:center;height:100vh;">
          <div style="background:white;padding:40px;border-radius:14px;box-shadow:0 4px 12px rgba(0,0,0,0.1);text-align:center;max-width:420px;">
            <h2 style="color:#1565c0;margin:0 0 10px;">🔒 Login Required</h2>
            <p style="color:#555;">You must be logged in to view your full scan report.</p>
            <a href="#" onclick="window.close()" style="display:inline-block;margin-top:12px;background:#1976d2;color:#fff;padding:8px 12px;border-radius:8px;text-decoration:none;font-weight:600;">Close</a>
          </div>
        </body></html>
        """
        return html, 401

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    return send_from_directory(templates_dir, "full-report.html")

# ==============================================================
# MAIN ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
