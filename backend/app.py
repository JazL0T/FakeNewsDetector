# ==============================================================
# Fake News Detector API (Final Secure + History + Full Report)
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

# ---------------- CONFIG ----------------
load_dotenv()
app = Flask(__name__)
CORS(app, origins=["*"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "users.db"))

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model2.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl")

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------- MODEL ----------------
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Model load failed: {e}")
    model, vectorizer = None, None

# ---------------- DATABASE ----------------
def init_db():
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """)
        conn.execute("""
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

init_db()

def get_db():
    return sqlite3.connect(app.config["DB_PATH"])

# ---------------- HELPERS ----------------
def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except:
        return None

def safe_json(data):
    return jsonify(json.loads(json.dumps(data, default=str)))

def tokenize_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

def predict_fake_news(text):
    if not model or not vectorizer:
        return {"error": "Model not loaded"}
    features = vectorizer.transform([tokenize_text(text)])
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return {"prediction": str(prediction), "confidence": float(max(probs))}

def analyze_text_heuristics(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * uppercase_ratio + 0.3 * min(exclamations / 5, 1)
    return {"sentiment": sentiment, "uppercase_ratio": uppercase_ratio, "exclamations": exclamations, "fake_score": fake_score}

def compute_trustability(url):
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted = ["bbc.com", "reuters.com", "nytimes.com", "apnews.com", "theguardian.com"]
    low = ["clickbait", "buzz", "rumor", "gossip", ".info", ".buzz", ".click"]
    if any(x in domain for x in trusted):
        return {"domain": domain, "trust_score": 90, "category": "Trusted"}
    elif any(x in domain for x in low):
        return {"domain": domain, "trust_score": 30, "category": "Suspicious"}
    return {"domain": domain, "trust_score": 60, "category": "Uncertain"}

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API running ✅"})

# --- AUTH ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400
    hashed = generate_password_hash(password)
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
        return jsonify({"message": "User registered"})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode({"username": username, "exp": datetime.utcnow() + timedelta(hours=2)}, app.config["JWT_SECRET"], algorithm="HS256")
    return jsonify({"token": token, "username": username})

# --- PREDICT ---
@app.route("/predict", methods=["POST"])
def predict():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    data = request.get_json() or {}
    text, headline, url = data.get("text", ""), data.get("headline", ""), data.get("url", "")
    if not text:
        return jsonify({"error": "Missing text"}), 400
    ml = predict_fake_news(text)
    heur = analyze_text_heuristics(text)
    trust = compute_trustability(url)
    if username:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (username, headline, url, text, ml["prediction"], ml["confidence"], json.dumps(heur), json.dumps(trust)))
            conn.commit()
    return safe_json({
        "username": username or "Guest",
        "headline": headline,
        "url": url,
        "prediction": "Fake" if ml["prediction"] == "0" else "Real",
        "confidence": ml["confidence"],
        "heuristics": heur,
        "trustability": trust
    })

# --- HISTORY ---
@app.route("/get-history", methods=["GET"])
def get_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, headline, url, prediction, confidence, timestamp FROM scans WHERE username=? ORDER BY timestamp DESC", (username,))
        rows = cur.fetchall()
    history = [{"id": r[0], "headline": r[1], "url": r[2], "prediction": r[3], "confidence": r[4], "timestamp": r[5]} for r in rows]
    return jsonify({"history": history})

# --- REPORT ---
@app.route("/get-report/<int:scan_id>", methods=["GET"])
def get_report(scan_id):
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return "<h3>Unauthorized - Please log in first</h3>", 401
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT headline, url, text, prediction, confidence, heuristics, trustability, timestamp FROM scans WHERE id=? AND username=?", (scan_id, username))
        row = cur.fetchone()
    if not row:
        return "<h3>Report not found</h3>", 404
    heur = json.loads(row[5])
    trust = json.loads(row[6])
    return render_template_string("""
    <html><head><title>Full Report</title>
    <style>
      body { font-family: Arial; background:#f5f9ff; margin:40px; color:#333; }
      h2 { color:#1565c0; }
      .box { background:white; padding:25px; border-radius:10px; box-shadow:0 4px 10px rgba(0,0,0,0.1); }
      .pred { font-weight:600; font-size:20px; color:{{'red' if row[3]=='0' else 'green'}}; }
    </style></head><body>
      <div class="box">
        <h2>📰 {{ row[0] or 'No Headline' }}</h2>
        <p><b>URL:</b> <a href="{{ row[1] }}" target="_blank">{{ row[1] }}</a></p>
        <p class="pred">Prediction: {{ 'Fake' if row[3]=='0' else 'Real' }} (Confidence: {{ '%.1f'|format(row[4]*100) }}%)</p>
        <hr>
        <h3>Heuristic Analysis</h3>
        <ul>
          <li>Sentiment: {{ heur.sentiment }}</li>
          <li>Uppercase Ratio: {{ heur.uppercase_ratio }}</li>
          <li>Exclamations: {{ heur.exclamations }}</li>
          <li>Fake Score: {{ heur.fake_score }}</li>
        </ul>
        <hr>
        <h3>Trustability</h3>
        <ul>
          <li>Domain: {{ trust.domain }}</li>
          <li>Trust Score: {{ trust.trust_score }}</li>
          <li>Category: {{ trust.category }}</li>
        </ul>
        <hr>
        <h3>Scanned Text</h3>
        <div style="background:#eef4ff;padding:10px;border-radius:6px;white-space:pre-wrap;">{{ row[2] }}</div>
      </div>
    </body></html>
    """, row=row, heur=heur, trust=trust)

# --- MAIN ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
