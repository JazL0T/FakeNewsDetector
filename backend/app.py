# ==============================================================
#  Fake News Detector 101 — Optimized Explainable AI API
#  Version: Render-Ready (2025) + EN/MY Dual Model + Top 4 Upgrades
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
from langdetect import detect, LangDetectException

# ============================================================== #
# CONFIGURATION
# ============================================================== #
load_dotenv()

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app, origins=["*"])

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "supersecretkey")
app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "jwt_secret")
app.config["DB_PATH"] = os.getenv("DB_PATH", os.path.join(BASE_DIR, "users.db"))

# EN (global/English) model (existing)
MODEL_PATH_EN = os.getenv("MODEL_PATH", os.path.join(MODELS_DIR, "model2.pkl"))
VECTORIZER_PATH_EN = os.getenv("VECTORIZER_PATH", os.path.join(MODELS_DIR, "vectorizer2.pkl"))

# MY (Malay) model (trained by your new script)
MODEL_PATH_MY = os.getenv("MALAY_MODEL_PATH", os.path.join(MODELS_DIR, "malay_model.pkl"))
VECTORIZER_PATH_MY = os.getenv("MALAY_VECTORIZER_PATH", os.path.join(MODELS_DIR, "malay_vectorizer.pkl"))

# ============================================================== #
# LOGGING
# ============================================================== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============================================================== #
# RATE LIMITING
# ============================================================== #
limiter = Limiter(get_remote_address, app=app, default_limits=["50 per minute"])
logging.info("🛡️ Rate limiting: 50 req/min per client")

# ============================================================== #
# MODELS (Lazy load both EN and MY)
# ============================================================== #
# EN
_en_model = _en_vectorizer = _en_coef = None
# MY
_my_model = _my_vectorizer = _my_coef = None

_model_lock = threading.Lock()

def _load_pair(model_path, vec_path):
    m = joblib.load(model_path)
    v = joblib.load(vec_path)
    coef = m.coef_[0] if hasattr(m, "coef_") else None
    return m, v, coef

def load_models_if_needed(lang: str):
    """
    lang: 'en' or 'ms'
    """
    global _en_model, _en_vectorizer, _en_coef
    global _my_model, _my_vectorizer, _my_coef

    with _model_lock:
        if lang == "en":
            if _en_model is None or _en_vectorizer is None:
                t0 = time.time()
                _en_model, _en_vectorizer, _en_coef = _load_pair(MODEL_PATH_EN, VECTORIZER_PATH_EN)
                logging.info(f"✅ EN model loaded in {time.time()-t0:.2f}s")
        else:  # 'ms'
            if _my_model is None or _my_vectorizer is None:
                if not (os.path.exists(MODEL_PATH_MY) and os.path.exists(VECTORIZER_PATH_MY)):
                    # If Malay model missing, we won't fail — we'll fallback to EN later
                    logging.warning("⚠️ Malay model/vectorizer not found — will fallback to EN.")
                    return
                t0 = time.time()
                _my_model, _my_vectorizer, _my_coef = _load_pair(MODEL_PATH_MY, VECTORIZER_PATH_MY)
                logging.info(f"✅ MY model loaded in {time.time()-t0:.2f}s")

# Warm EN model in background (fastest route for most users)
def preload_model_async():
    def _load():
        try:
            load_models_if_needed("en")
            logging.info("🚀 Background EN model preload complete.")
        except Exception as e:
            logging.error(f"⚠️ Background preload failed: {e}")
    threading.Thread(target=_load, daemon=True).start()

preload_model_async()

# ============================================================== #
# DATABASE (SQLite WAL + Retry)
# ============================================================== #
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

# ============================================================== #
# TEXT & PREDICTION UTILITIES
# ============================================================== #
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)  # Malay uses Latin script; OK
    return text.lower().strip()

def detect_lang(text: str) -> str:
    try:
        code = detect(text)
        return "ms" if code == "ms" else "en"
    except LangDetectException:
        return "en"

@lru_cache(maxsize=512)
def _cached_predict(lang: str, clean_text_hash: str, original_text: str):
    """
    Language-aware cached prediction. Uses the right model/vectorizer.
    Falls back to EN if MY model is missing.
    """
    # Ensure model is loaded
    load_models_if_needed(lang)

    # Choose pair
    if lang == "ms" and _my_model and _my_vectorizer:
        model, vec = _my_model, _my_vectorizer
        model_used = "malay"
    else:
        model, vec = _en_model, _en_vectorizer
        model_used = "english"

    features = vec.transform([original_text])
    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        conf = float(max(probs))
        class_probs = {"0": float(probs[0]), "1": float(probs[1])}
    else:
        # Approx confidence if no proba
        if hasattr(model, "decision_function"):
            df = model.decision_function(features)
            conf = float(1 / (1 + math.exp(-abs(df[0]))))
        else:
            conf = 0.5
        class_probs = {"0": 1 - conf, "1": conf}

    return {"prediction": str(pred), "confidence": conf, "class_probs": class_probs, "model_used": model_used}

def predict_fake_news(text: str) -> dict:
    if not text.strip():
        return {"error": "Empty text"}
    lang = detect_lang(text)
    text_hash = hashlib.sha256(tokenize_text(text).encode()).hexdigest()
    out = _cached_predict(lang, text_hash, text)
    out["language"] = "Malay" if lang == "ms" else "English"
    return out, (lang == "ms" and _my_model and _my_vectorizer)

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

# ============================================================== #
# TRUSTABILITY (Cached) — includes Malaysia-aware categories
# ============================================================== #
@lru_cache(maxsize=400)
def compute_trustability(url: str) -> dict:
    domain = tldextract.extract(url).registered_domain or "unknown"

    trusted_global = [
        "bbc.com","reuters.com","apnews.com","associatedpress.com","nytimes.com","theguardian.com",
        "cnn.com","npr.org","bloomberg.com","washingtonpost.com","aljazeera.com","forbes.com",
        "cnbc.com","dw.com","theatlantic.com","axios.com","politico.com","time.com","economist.com",
        "usatoday.com","abcnews.go.com","nbcnews.com","cbsnews.com","wsj.com","ft.com",
        "investopedia.com","marketwatch.com","morningstar.com","businessinsider.com",
        "nature.com","sciencedaily.com","scientificamerican.com","nationalgeographic.com",
        "newscientist.com","space.com","theconversation.com","arstechnica.com","techcrunch.com",
        "wired.com","engadget.com","euronews.com","bbc.co.uk","guardian.co.uk","spiegel.de",
        "tagesschau.de","repubblica.it","elpais.com","lemonde.fr","scmp.com","nikkei.com",
        "abc.net.au","sbs.com.au","enca.com","news24.com","bbcafrica.com","theafricareport.com",
        "arabnews.com","thenationalnews.com","clarin.com","folha.uol.com.br","lanacion.com.ar","g1.globo.com"
    ]
    trusted_my = [
        "thestar.com.my","malaymail.com","bernama.com","astroawani.com",
        "themalaysianreserve.com","freemalaysiatoday.com","theborneopost.com","theedgemalaysia.com"
    ]
    suspicious = [
        "clickbait","rumor","gossip",".buzz",".click","wordpress","blogspot",
        "viralnews","trendingnow","celebrityleak","beforeitsnews.com","thegatewaypundit.com",
        "infowars.com","naturalnews.com","breitbart.com","sputniknews.com","rt.com",
        "zerohedge.com","theblaze.com","dailycaller.com","theepochtimes.com"
    ]

    if any(t in domain for t in trusted_my):
        return {"domain": domain, "trust_score": 90, "category": "Trusted (Malaysia)"}
    if any(t in domain for t in trusted_global):
        return {"domain": domain, "trust_score": 90, "category": "Trusted"}
    if any(s in domain for s in suspicious):
        return {"domain": domain, "trust_score": 30, "category": "Suspicious"}
    if ".my" in domain:
        return {"domain": domain, "trust_score": 60, "category": "Unverified Malaysian Source"}
    return {"domain": domain, "trust_score": 50, "category": "Uncertain"}

# ============================================================== #
# JWT & RESPONSE HELPERS
# ============================================================== #
def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None

def safe_json(data):
    return jsonify(json.loads(json.dumps(data, default=str)))

# ============================================================== #
# EXPLAINABILITY (uses the ACTIVE model/vectorizer)
# ============================================================== #
FAKE_KEYWORDS = {"shocking","exclusive","miracle","hoax","exposed","click here","secret","scam","rumor"}
REAL_KEYWORDS = {"official","research","confirmed","report","sources","data","analysis","statement","evidence"}

def keyword_hits(line):
    l = line.lower()
    return [w for w in FAKE_KEYWORDS if w in l], [w for w in REAL_KEYWORDS if w in l]

def tfidf_line_score(line, model, vectorizer, coef_vector):
    if model is None or vectorizer is None or coef_vector is None:
        return 0.0
    score = 0.0
    if hasattr(vectorizer, "vocabulary_"):
        for word in tokenize_text(line).split():
            idx = vectorizer.vocabulary_.get(word)
            if idx is not None and idx < len(coef_vector):
                score += float(coef_vector[idx])
    return score

def line_heuristic_score(line):
    s = TextBlob(line).sentiment.polarity
    excls = line.count("!")
    words = line.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    return 0.5 * (1 - abs(s)) + 0.3 * upper_ratio + 0.2 * min(excls / 3, 1)

def explain_text(text, trust, final_pred, model, vectorizer, coef_vector):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    highlighted, reasons = [], []
    reasons.append(f"Domain '{trust['domain']}' marked as {trust['category']}.")

    for line in lines:
        w_tfidf = tfidf_line_score(line, model, vectorizer, coef_vector)
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

    # sort strongest contributors first (optional UI)
    highlighted.sort(key=lambda d: abs(d["weight"]), reverse=True)
    return highlighted, reasons

# ============================================================== #
# ROUTES
# ============================================================== #
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector 101 API ✅"})

@app.route("/predict", methods=["POST"])
@limiter.limit("10 per minute")
def predict():
    try:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)
        data = request.get_json() or {}

        text = data.get("text", "")
        headline = data.get("headline", "")
        url = data.get("url", "")
        if not text:
            return jsonify({"error": "Missing text"}), 400

        # Predict (language-aware)
        ml, used_malay = predict_fake_news(text)
        final_label = "Fake" if ml["prediction"] == "0" else "Real"

        # Pick the active model/vectorizer for explainability
        if used_malay and _my_model and _my_vectorizer:
            model, vectorizer, coef = _my_model, _my_vectorizer, _my_coef
        else:
            model, vectorizer, coef = _en_model, _en_vectorizer, _en_coef

        heur = analyze_text_heuristics(text)
        trust = compute_trustability(url)
        lines, reasons = explain_text(text, trust, final_label, model, vectorizer, coef)

        # Save (if logged-in)
        if username:
            def _save():
                with get_db_connection() as conn:
                    conn.execute("""
                        INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (username, headline, url, text, ml["prediction"], ml["confidence"],
                          json.dumps(heur), json.dumps(trust)))
                    conn.commit()
            with_retry(_save)

        return safe_json({
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "language": ml["language"],
            "model_used": ml["model_used"],  # 'english' or 'malay'
            "prediction": final_label,
            "confidence": ml["confidence"],
            "class_probs": ml["class_probs"],
            "heuristics": heur,
            "trustability": trust,
            "explain": {"lines": lines[:50], "reasons": reasons}
        })

    except Exception as e:
        logging.exception(f"❌ Prediction failed")
        return jsonify({"error": "Model prediction failed. Please try again later."}), 500

# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Running locally on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
