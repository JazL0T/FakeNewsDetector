# ==============================================================
#  Fake News Detector API (FINAL - Full Report + Auth Fix)
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json
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
# DATABASE SETUP
# ============================================================== #
def init_db():
    """Initialize database and create tables."""
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                headline TEXT,
                url TEXT,
                text TEXT,
                prediction TEXT,
                confidence REAL,
                heuristics TEXT,
                trustability TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_db_connection():
    return sqlite3.connect(app.config["DB_PATH"])

init_db()

# ============================================================== #
# HELPERS
# ============================================================== #
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
    fake_score = (0.4 * (1 - abs(sentiment)) +
                  0.3 * uppercase_ratio +
                  0.3 * min(exclamations / 5, 1))
    return {
        "sentiment": sentiment,
        "uppercase_ratio": uppercase_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score
    }

def compute_trustability(url: str) -> dict:
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted_sources = ["bbc.com", "reuters.com", "apnews.com", "nytimes.com", "theguardian.com", "npr.org"]
    suspicious_markers = ["clickbait", "rumor", "gossip", "unknownblog", ".info", ".buzz", ".click"]

    score = 50
    if any(src in domain for src in trusted_sources):
        score, category = 90, "Trusted"
    elif any(m in domain for m in suspicious_markers):
        score, category = 30, "Suspicious"
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

# ============================================================== #
# ROUTES
# ============================================================== #

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API running ✅"})

# ---------- AUTH ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    # 🔒 Enforce minimum password length (e.g., 8 characters)
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters long."}), 400

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
        "exp": datetime.utcnow() + timedelta(hours=3)
    }, app.config["JWT_SECRET"], algorithm="HS256")
    return jsonify({"token": token, "username": username})

# ---------- PREDICTION ----------
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
    heur = analyze_text_heuristics(text)
    trust = compute_trustability(url)

    if username:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, headline, url, text,
                  ml["prediction"], ml["confidence"],
                  json.dumps(heur), json.dumps(trust)))
            conn.commit()

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

# ---------- HISTORY ----------
@app.route("/get-history", methods=["GET"])
def get_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, headline, url, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE username = ? ORDER BY timestamp DESC
        """, (username,))
        rows = cur.fetchall()
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "headline": r[1],
            "url": r[2],
            "prediction": "Fake" if r[3] == "0" else "Real",
            "confidence": r[4],
            "heuristics": json.loads(r[5]),
            "trustability": json.loads(r[6]),
            "timestamp": r[7]
        })
    return jsonify({"history": history})

# ---------- FULL REPORT ----------
@app.route("/get-report/<int:scan_id>")
def get_report(scan_id):
    # ✅ Chrome-safe: read token from ?token=
    token = request.args.get("token")
    if not token:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")

    username = verify_jwt(token)
    if not username:
        return "<h3>Unauthorized - Please log in first.</h3>", 401

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, headline, url, text, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE id = ? AND username = ?
        """, (scan_id, username))
        row = cur.fetchone()

    if not row:
        return "<h3>Report not found or not owned by you.</h3>", 404

    heur = json.loads(row[6])
    trust = json.loads(row[7])

    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>Full Report - Fake News Detector</title>
      <style>
        body { font-family: Arial; background:#f7f9fc; color:#333; padding:40px; }
        .container { background:white; max-width:800px; margin:auto; padding:30px; border-radius:12px;
                     box-shadow:0 4px 10px rgba(0,0,0,0.1); }
        h1 { color:#1565c0; text-align:center; }
        .fake { color:#e53935; font-weight:600; }
        .real { color:#2e7d32; font-weight:600; }
        hr { border: none; height: 1px; background:#ddd; margin:20px 0; }
        .section { margin-bottom:20px; }
        .data { background:#eef4ff; padding:12px; border-radius:8px; white-space:pre-wrap; }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Fake News Detector - Full Report</h1>
        <p><b>Headline:</b> {{ row[1] or "(No headline)" }}</p>
        <p><b>URL:</b> <a href="{{ row[2] }}" target="_blank">{{ row[2] }}</a></p>
        <p><b>Scanned by:</b> {{ username }}</p>
        <p><b>Date:</b> {{ row[8] }}</p>
        <p class="{{ 'fake' if row[4]=='0' else 'real' }}">
          Prediction: {{ 'Fake' if row[4]=='0' else 'Real' }}
          (Confidence: {{ '%.1f'|format(row[5]*100) }}%)
        </p>
        <hr>

        <div class="section">
          <h3>Heuristic Analysis</h3>
          <ul>
            <li>Sentiment: {{ heur.sentiment }}</li>
            <li>Uppercase Ratio: {{ heur.uppercase_ratio }}</li>
            <li>Exclamations: {{ heur.exclamations }}</li>
            <li>Fake Score: {{ heur.fake_score }}</li>
          </ul>
        </div>

        <div class="section">
          <h3>Trustability</h3>
          <ul>
            <li>Domain: {{ trust.domain }}</li>
            <li>Trust Score: {{ trust.trust_score }}</li>
            <li>Category: {{ trust.category }}</li>
          </ul>
        </div>

        <div class="section">
          <h3>Scanned Text</h3>
          <div class="data">{{ row[3] }}</div>
        </div>
      </div>
    </body>
    </html>
    """, row=row, heur=heur, trust=trust, username=username)

# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
