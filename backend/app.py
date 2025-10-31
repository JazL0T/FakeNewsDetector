# ==============================================================
#  Fake News Detector API (Render-ready v3)
#  - Keeps all previous features
#  - Adds: Website Trustability Analysis
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, jwt, socket, requests
from datetime import datetime, timedelta
from urllib.parse import urlparse
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv

# ===== CONFIG =====
BASE_DIR = os.path.dirname(__file__)
STATIC_DIR = os.path.join(BASE_DIR, "static")

load_dotenv()

# Ensure TextBlob corpora
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import textblob.download_corpora as download
    logging.info("⬇️ Downloading missing TextBlob corpora...")
    download.download_all()

# Flask App
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app, origins=["*"])

# Env Vars
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "supersecretkey"),
    JWT_SECRET=os.getenv("JWT_SECRET", "jwt_secret"),
    DB_PATH=os.getenv("DB_PATH", os.path.join(BASE_DIR, "users.db")),
)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models/model2.pkl"))
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", os.path.join(BASE_DIR, "models/vectorizer2.pkl"))

# ===== LOGGING =====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===== LOAD MODEL =====
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")
    model = vectorizer = None

# ===== DB SETUP =====
def init_db():
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
    conn = sqlite3.connect(app.config["DB_PATH"], check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ===== HELPERS =====
def safe_json(data): return jsonify(json.loads(json.dumps(data, default=str)))
def tokenize_text(text): return re.sub(r"[^a-zA-Z\s]", "", text.lower().strip())

def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

# ===== MACHINE LEARNING PREDICTION =====
def predict_fake_news(text):
    if not model or not vectorizer:
        return {"error": "Model not loaded"}
    features = vectorizer.transform([tokenize_text(text)])
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    return {
        "prediction": str(prediction),
        "confidence": float(max(probs)),
        "class_probs": {"0": float(probs[0]), "1": float(probs[1])},
    }

# ===== HEURISTICS =====
def analyze_text_heuristics(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
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

# ===== TRUSTABILITY ANALYSIS =====
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "npr.org", "bloomberg.com", "aljazeera.com",
    "washingtonpost.com", "forbes.com", "cnbc.com"
]
SUSPICIOUS_TLDS = [".info", ".click", ".buzz", ".ru", ".top", ".xyz", ".blogspot.com"]

def analyze_domain_trust(url: str):
    if not url:
        return {"domain": None, "trust_score": 0, "category": "Unknown"}

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().replace("www.", "")
        score = 50  # baseline

        # Known reputable domains
        if any(t in domain for t in TRUSTED_SOURCES):
            score += 40

        # Suspicious TLDs or patterns
        if any(domain.endswith(t) for t in SUSPICIOUS_TLDS):
            score -= 30

        # Domain age (optional)
        try:
            import whois
            w = whois.whois(domain)
            if isinstance(w.creation_date, datetime):
                age_years = (datetime.now() - w.creation_date).days / 365
                if age_years < 1:
                    score -= 10
                elif age_years > 5:
                    score += 5
        except Exception:
            pass  # WHOIS not critical

        # DNS check
        try:
            socket.gethostbyname(domain)
        except Exception:
            score -= 20

        # Bound score 0–100
        score = max(0, min(score, 100))

        # Label
        category = (
            "Trusted" if score >= 70 else
            "Uncertain" if score >= 40 else
            "Suspicious"
        )

        return {"domain": domain, "trust_score": score, "category": category}

    except Exception as e:
        logging.warning(f"Trust check error: {e}")
        return {"domain": None, "trust_score": 0, "category": "Unknown"}

# ===== ROUTES =====
@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector API running"})

@app.route("/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.4"})

# --- AUTH ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
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
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    with get_db_connection() as conn:
        row = conn.execute("SELECT password FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode({"username": username, "exp": datetime.utcnow() + timedelta(hours=2)},
                       app.config["JWT_SECRET"], algorithm="HS256")
    return jsonify({"token": token, "username": username})

# --- PREDICT (with Trustability) ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)

        data = request.get_json() or {}
        text, headline, url = data.get("text", ""), data.get("headline", ""), data.get("url", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        ml_result = predict_fake_news(text)
        heuristics = analyze_text_heuristics(text)
        trustability = analyze_domain_trust(url)

        # Sentence-level breakdown
        blob = TextBlob(text)
        sentences = []
        for sent in blob.sentences:
            stext = str(sent)
            spred = predict_fake_news(stext)
            if "error" in spred:
                continue
            sentences.append({
                "text": stext,
                "prediction": "Fake" if spred["prediction"] == "0" else "Real",
                "confidence": spred["confidence"]
            })

        result = {
            "username": username or "guest",
            "headline": headline,
            "url": url,
            "prediction": "Fake" if ml_result["prediction"] == "0" else "Real",
            "confidence": ml_result["confidence"],
            "heuristics": heuristics,
            "trustability": trustability,
            "sentences": sentences,
        }
        return safe_json(result)
    except Exception as e:
        logging.exception("❌ Prediction error:")
        return jsonify({"error": str(e)}), 500

# --- STATIC REPORT PAGE ---
@app.route("/full-report")
def full_report():
    return app.send_static_file("full-report.html")

# --- ERRORS ---
@app.errorhandler(404)
def nf(e): return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def ie(e):
    logging.exception("Internal error")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
