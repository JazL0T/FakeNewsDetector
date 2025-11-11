# ==============================================================
#  Fake News Detector 101 — Optimized Explainable AI API
#  Version: Render-Ready (2025) + Enhanced (Top 4 Upgrades)
# ==============================================================
# ✨ Key Improvements:
# 1. Background model preloading (faster cold start)
# 2. Graceful prediction error handling
# 3. Rate limiting (anti-abuse)
# 4. Cached domain trustability
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, math, time, threading, hashlib
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv
import tldextract
from functools import lru_cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ==============================================================
# CONFIGURATION
# ==============================================================
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, origins=["*"])

# Security & App Config
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv(
    "DB_PATH", os.path.join(os.path.dirname(__file__), "users.db")
)

MODEL_PATH = os.getenv(
    "MODEL_PATH", os.path.join(os.path.dirname(__file__), "models", "model2.pkl")
)
VECTORIZER_PATH = os.getenv(
    "VECTORIZER_PATH", os.path.join(os.path.dirname(__file__), "models", "vectorizer2.pkl")
)

# ==============================================================
# LOGGING
# ==============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ==============================================================
# RATE LIMITING (Flask-Limiter)
# ==============================================================
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["50 per minute"],  # Global default
)
logging.info("🛡️ Rate limiting enabled: 50 requests/minute per client.")

# ==============================================================
# MODEL (Lazy Load + Background Preload)
# ==============================================================
_model, _vectorizer, _coef_vector, _vocab = None, None, None, None
_model_lock = threading.Lock()

def load_model():
    """Load model and vectorizer only once (thread-safe)."""
    global _model, _vectorizer, _coef_vector, _vocab
    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    with _model_lock:
        if _model is None or _vectorizer is None:
            t0 = time.time()
            _model = joblib.load(MODEL_PATH)
            _vectorizer = joblib.load(VECTORIZER_PATH)
            if hasattr(_model, "coef_"):
                _coef_vector = _model.coef_[0]
                if hasattr(_vectorizer, "get_feature_names_out"):
                    _vocab = _vectorizer.get_feature_names_out()
            logging.info(f"✅ Model loaded in {time.time()-t0:.2f}s")
    return _model, _vectorizer

# 🔹 Background preload to warm up model at startup
def preload_model_async():
    def _load():
        try:
            load_model()
            logging.info("🚀 Background model preload complete.")
        except Exception as e:
            logging.error(f"⚠️ Background model preload failed: {e}")
    threading.Thread(target=_load, daemon=True).start()

# Start background preload
preload_model_async()

# ==============================================================
# DATABASE (SQLite WAL + Retry)
# ==============================================================
def init_db():
    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
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
    conn = sqlite3.connect(app.config["DB_PATH"], timeout=10, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn

def with_retry(fn, *args, **kwargs):
    for i in range(5):
        try:
            return fn(*args, **kwargs)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                time.sleep(0.2 * (i + 1))
                continue
            raise
    raise Exception("Database locked after retries.")

init_db()

# ==============================================================
# TEXT UTILITIES
# ==============================================================
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

@lru_cache(maxsize=512)
def cached_prediction(clean_text_hash: str, original_text: str):
    model, vectorizer = load_model()
    features = vectorizer.transform([tokenize_text(original_text)])
    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        conf = float(max(probs))
        class_probs = {"0": float(probs[0]), "1": float(probs[1])}
    else:
        conf = 0.5
        class_probs = {"0": 0.5, "1": 0.5}
    return {"prediction": str(pred), "confidence": conf, "class_probs": class_probs}

def predict_fake_news(text: str) -> dict:
    if not text.strip():
        return {"error": "Empty text"}
    text_hash = hashlib.sha256(tokenize_text(text).encode()).hexdigest()
    return cached_prediction(text_hash, text)

def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * upper_ratio + 0.3 * min(exclamations / 5, 1)
    return {
        "sentiment": sentiment,
        "uppercase_ratio": upper_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score
    }

# ==============================================================
# PERFORMANCE BOOST — Cached Domain Trustability
# ==============================================================
@lru_cache(maxsize=200)
def compute_trustability(url: str) -> dict:
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted = [
    # --- International Mainstream Media ---
    "bbc.com", "reuters.com", "apnews.com", "associatedpress.com",
    "nytimes.com", "theguardian.com", "cnn.com", "npr.org",
    "bloomberg.com", "washingtonpost.com", "aljazeera.com",
    "forbes.com", "cnbc.com", "dw.com", "theatlantic.com",
    "axios.com", "politico.com", "time.com", "economist.com",
    "usatoday.com", "abcnews.go.com", "nbcnews.com", "cbsnews.com",

    # --- Fact-checking and Verification Organizations ---
    "snopes.com", "factcheck.org", "politifact.com", "afp.com",
    "fullfact.org", "africacheck.org", "poynter.org", "boomlive.in",
    "maldita.es", "verafiles.org", "truthorfiction.com",
    "leadstories.com", "checkyourfact.com", "euvsdisinfo.eu",

    # --- Science, Research & Tech News ---
    "nature.com", "sciencedaily.com", "scientificamerican.com",
    "nationalgeographic.com", "newscientist.com", "space.com",
    "theconversation.com", "arstechnica.com", "techcrunch.com",
    "wired.com", "engadget.com",

    # --- Financial / Economic Outlets ---
    "wsj.com", "ft.com", "investopedia.com", "marketwatch.com",
    "morningstar.com", "businessinsider.com",

    # --- Asia-Pacific / Regional Trusted News ---
    "straitstimes.com", "channelnewsasia.com", "themalaysianreserve.com",
    "thestar.com.my", "malaymail.com", "bernama.com", "nikkei.com",
    "japantimes.co.jp", "scmp.com", "abc.net.au", "sbs.com.au",

    # --- European Trusted Outlets ---
    "lemonde.fr", "euronews.com", "bbc.co.uk", "guardian.co.uk",
    "spiegel.de", "tagesschau.de", "repubblica.it", "elpais.com",

    # --- African & Middle East Trusted Outlets ---
    "enca.com", "news24.com", "bbcafrica.com", "theafricareport.com",
    "arabnews.com", "thenationalnews.com",

    # --- Latin American Trusted Outlets ---
    "bbc.com/mundo", "clarin.com", "folha.uol.com.br", "elpais.com",
    "lanacion.com.ar", "g1.globo.com"
]
    
    suspicious = [
    "clickbait", "rumor", "gossip", ".buzz", ".click",
    "wordpress", "blogspot", "viralnews", "trendingnow",
    "celebrityleak", "beforeitsnews.com", "thegatewaypundit.com",
    "infowars.com", "naturalnews.com", "breitbart.com",
    "sputniknews.com", "rt.com", "zerohedge.com",
    "theblaze.com", "dailycaller.com", "theepochtimes.com"
]

    if any(t in domain for t in trusted):
        return {"domain": domain, "trust_score": 90, "category": "Trusted"}
    elif any(s in domain for s in suspicious):
        return {"domain": domain, "trust_score": 30, "category": "Suspicious"}
    else:
        return {"domain": domain, "trust_score": 50, "category": "Uncertain"}

# ==============================================================
# JWT & RESPONSE HELPERS
# ==============================================================
def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def safe_json(data):
    return jsonify(json.loads(json.dumps(data, default=str)))

# ==============================================================
# EXPLAINABILITY ENGINE
# ==============================================================
FAKE_KEYWORDS = {"shocking", "exclusive", "miracle", "hoax", "exposed", "click here", "secret"}
REAL_KEYWORDS = {"official", "research", "confirmed", "report", "sources", "data", "analysis"}

def keyword_hits(line):
    l = line.lower()
    return [w for w in FAKE_KEYWORDS if w in l], [w for w in REAL_KEYWORDS if w in l]

def tfidf_line_score(line):
    if _model is None:
        load_model()
    score = 0.0
    if hasattr(_vectorizer, "vocabulary_") and hasattr(_model, "coef_"):
        for word in tokenize_text(line).split():
            idx = _vectorizer.vocabulary_.get(word)
            if idx is not None and idx < len(_model.coef_[0]):
                score += float(_model.coef_[0][idx])
    return score

def line_heuristic_score(line):
    s = TextBlob(line).sentiment.polarity
    excls = line.count("!")
    upper_ratio = sum(1 for w in line.split() if w.isupper()) / max(1, len(line.split()))
    return 0.5 * (1 - abs(s)) + 0.3 * upper_ratio + 0.2 * min(excls / 3, 1)

def explain_text(text, trust, final_pred):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    highlighted, reasons = [], []
    reasons.append(f"Domain '{trust['domain']}' marked as {trust['category']}.")
    for line in lines:
        w_tfidf = tfidf_line_score(line)
        fp = line_heuristic_score(line)
        w_heur = (0.5 - fp)
        fake_hits, real_hits = keyword_hits(line)
        kw_signal = 0.0
        if fake_hits: kw_signal -= 0.3 * len(fake_hits)
        if real_hits: kw_signal += 0.2 * len(real_hits)
        total = w_tfidf + w_heur + kw_signal
        tags = []
        if fake_hits: tags.append(f"Fake cues: {', '.join(fake_hits)}")
        if real_hits: tags.append(f"Real cues: {', '.join(real_hits)}")
        if abs(w_tfidf) > 0.2: tags.append("Model-weighted term influence")
        highlighted.append({"text": line, "weight": total, "tags": tags})
    if final_pred == "Fake":
        reasons.append("Model detected sensational or biased tone.")
    elif final_pred == "Real":
        reasons.append("Model detected balanced and factual language.")
    return highlighted, reasons

# ==============================================================
# ROUTES
# ==============================================================
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector 101 API ✅"})

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")  # 10 predictions per minute per IP
def predict():
    """Main prediction route with error handling."""
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)
        data = request.get_json() or {}
        text, headline, url = data.get("text", ""), data.get("headline", ""), data.get("url", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        ml = predict_fake_news(text)
        if "error" in ml:
            raise ValueError(ml["error"])

        heur = analyze_text_heuristics(text)
        trust = compute_trustability(url)
        final_label = "Fake" if ml["prediction"] == "0" else "Real"
        lines, reasons = explain_text(text, trust, final_label)

        # Save history (if logged in)
        if username:
            def _save():
                with get_db_connection() as conn:
                    conn.execute("""
                        INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (username, headline, url, text, ml["prediction"], ml["confidence"], json.dumps(heur), json.dumps(trust)))
                    conn.commit()
            with_retry(_save)

        return safe_json({
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "prediction": final_label,
            "confidence": ml["confidence"],
            "class_probs": ml["class_probs"],
            "heuristics": heur,
            "trustability": trust,
            "explain": {"lines": lines[:50], "reasons": reasons}
        })

    except Exception as e:
        logging.error(f"❌ Prediction failed: {e}")
        return jsonify({"error": "Model prediction failed. Please try again later."}), 500

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Running locally on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
