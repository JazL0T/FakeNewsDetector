# ==============================================================
#  Fake News Detector 101 — Optimized Explainable AI API
#  Version: Render-Ready (2025.11 FINAL - FIXED)
#  Features: Dual EN/MY Models (Malay text only) + Auth + History + Explainability
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=4)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, time, threading, hashlib, math  # ✅ Added math here
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv
import tldextract
from functools import lru_cache
from flask_limiter import Limiter
from langdetect import detect, LangDetectException
from logging.handlers import RotatingFileHandler

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

MODEL_PATH_EN = os.path.join(MODELS_DIR, "model2.pkl")
VECTORIZER_PATH_EN = os.path.join(MODELS_DIR, "vectorizer2.pkl")
MODEL_PATH_MY = os.path.join(MODELS_DIR, "malay_model.pkl")
VECTORIZER_PATH_MY = os.path.join(MODELS_DIR, "malay_vectorizer.pkl")

# ============================================================== #
# LOGGING
# ============================================================== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ✅ Rotating Log File (saves 3 backups, 2MB each)
log_handler = RotatingFileHandler("app.log", maxBytes=2_000_000, backupCount=3)
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
log_handler.setFormatter(log_formatter)
app.logger.addHandler(log_handler)
logging.getLogger().addHandler(log_handler)

# ============================================================== #
# RATE LIMITING
# ============================================================== #
def get_user_or_ip():
    """Limit logged-in users by username; guests by IP."""
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        username = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"]).get("username")
    except Exception:
        username = None
    return username or request.remote_addr

limiter = Limiter(get_user_or_ip, app=app, default_limits=["100 per 20 minutes"])
PREDICT_LIMIT = "20 per 20 minutes"
logging.info("🛡️ Rate limiting enabled: 20 predictions per 20 minutes")

# ============================================================== #
# MODEL LOADING (Lazy + Background warmup)
# ============================================================== #
_en_model = _en_vectorizer = _en_coef = None
_my_model = _my_vectorizer = _my_coef = None
_model_lock = threading.Lock()

def _load_pair(model_path, vec_path):
    """Load model and vectorizer safely."""
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        raise FileNotFoundError(f"Missing model/vectorizer: {model_path}")
    m = joblib.load(model_path)
    v = joblib.load(vec_path)
    coef = m.coef_[0] if hasattr(m, "coef_") else None
    return m, v, coef

def load_models_if_needed(lang: str):
    global _en_model, _en_vectorizer, _en_coef
    global _my_model, _my_vectorizer, _my_coef
    with _model_lock:
        try:
            if lang == "en":
                if _en_model is None or _en_vectorizer is None:
                    t0 = time.time()
                    _en_model, _en_vectorizer, _en_coef = _load_pair(MODEL_PATH_EN, VECTORIZER_PATH_EN)
                    logging.info(f"✅ English model loaded in {time.time()-t0:.2f}s")
            elif lang == "ms":
                if _my_model is None or _my_vectorizer is None:
                    t0 = time.time()
                    _my_model, _my_vectorizer, _my_coef = _load_pair(MODEL_PATH_MY, VECTORIZER_PATH_MY)
                    logging.info(f"✅ Malay model loaded in {time.time()-t0:.2f}s")
        except Exception as e:
            logging.warning(f"⚠️ Model load failed for {lang}: {e}")
            if lang == "ms":
                logging.info("🔁 Fallback to English model.")
                load_models_if_needed("en")

# preload English model in background
threading.Thread(target=lambda: load_models_if_needed("en"), daemon=True).start()

# ============================================================== #
# DATABASE
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
                language TEXT,
                risk_level TEXT,
                runtime REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_db_connection():
    """Optimized SQLite connection with better performance and safety."""
    conn = sqlite3.connect(app.config["DB_PATH"], timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # ✅ Return dict-like rows instead of tuples
    conn.execute("PRAGMA busy_timeout = 10000;")
    conn.execute("PRAGMA cache_size = 10000;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn

init_db()

# ============================================================== #
# UTILITIES
# ============================================================== #
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower().strip()

def detect_lang(text: str) -> str:
    """Detect Malay only if text language is Malay or contains Malay words."""
    try:
        code = detect(text)
    except LangDetectException:
        code = "en"

    text_lower = text.lower()
    malay_keywords = [
        "kerajaan","rakyat","berita","politik","bantuan",
        "negara","negeri","parlimen","menteri","malaysia","sabahan"
    ]

    if code == "ms" or any(w in text_lower for w in malay_keywords):
        return "ms"
    return "en"

@lru_cache(maxsize=512)
def _cached_predict(lang: str, clean_text_hash: str, original_text: str):
    load_models_if_needed(lang)
    model, vec, coef, used = (
        (_my_model, _my_vectorizer, _my_coef, "Malay")
        if lang == "ms" and _my_model and _my_vectorizer
        else (_en_model, _en_vectorizer, _en_coef, "English")
    )

    features = vec.transform([original_text])
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
    conf = float(max(probs))
    return {
        "prediction": str(pred),
        "confidence": conf,
        "class_probs": {"0": float(probs[0]), "1": float(probs[1])},
        "model_used": used,
        "coef_vector": coef
    }

def predict_fake_news(text: str):
    """
    Perform AI-based fake news prediction with support for long articles.
    - Short texts: cached for faster results.
    - Long texts: analyzed in multiple segments for accuracy and stability.
    """
    if not text.strip():
        return {"error": "Empty text"}

    # --- Language detection ---
    lang = detect_lang(text)
    load_models_if_needed(lang)
    model, vec, coef, used = (
        (_my_model, _my_vectorizer, _my_coef, "Malay")
        if lang == "ms" and _my_model and _my_vectorizer
        else (_en_model, _en_vectorizer, _en_coef, "English")
    )

    # --- Prepare text and hash for caching ---
    clean_text = tokenize_text(text)
    text_hash = hashlib.sha256(clean_text.encode()).hexdigest()
    word_count = len(clean_text.split())

    # --- For short text (<700 words): use cache ---
    if word_count < 700:
        res = _cached_predict(lang, text_hash, text)
        res["language"] = "Malay" if lang == "ms" else "English"
        res["chunks_analyzed"] = 1
        logging.info(f"🧠 Cached prediction used for short text ({word_count} words).")
        return res, (lang == "ms" and _my_model and _my_vectorizer)

    # --- For long text: split into manageable chunks ---
    def chunk_text(text, max_words=700):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current, word_count = [], [], 0
        for sent in sentences:
            w = len(sent.split())
            if word_count + w > max_words and current:
                chunks.append(" ".join(current))
                current, word_count = [sent], w
            else:
                current.append(sent)
                word_count += w
        if current:
            chunks.append(" ".join(current))
        return chunks

    chunks = chunk_text(text)
    chunk_preds, chunk_probs = [], []

    # --- Evaluate each chunk individually ---
    for chunk in chunks:
        try:
            features = vec.transform([chunk])
            pred = model.predict(features)[0]
            probs = (
                model.predict_proba(features)[0]
                if hasattr(model, "predict_proba")
                else [0.5, 0.5]
            )
            chunk_preds.append(pred)
            chunk_probs.append(probs)
        except Exception as e:
            logging.warning(f"⚠️ Chunk skipped: {e}")

    # --- Aggregate results across chunks ---
    if not chunk_preds:
        return {"error": "No valid text processed"}

    avg_probs = [
        sum(p[i] for p in chunk_probs) / len(chunk_probs)
        for i in range(2)
    ]
    avg_conf = float(max(avg_probs))
    final_pred = int(avg_probs[1] > avg_probs[0])

    result = {
        "prediction": str(final_pred),
        "confidence": avg_conf,
        "class_probs": {"0": float(avg_probs[0]), "1": float(avg_probs[1])},
        "model_used": used,
        "coef_vector": coef,
        "language": "Malay" if lang == "ms" else "English",
        "chunks_analyzed": len(chunks)
    }

    logging.info(
        f"🧠 Long article detected ({word_count} words) | "
        f"{len(chunks)} segments analyzed | Language: {result['language']}"
    )

    return result, (lang == "ms" and _my_model and _my_vectorizer)

def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * upper_ratio + 0.3 * min(exclamations / 5, 1)
    return {"sentiment": sentiment, "uppercase_ratio": upper_ratio, "exclamations": exclamations, "fake_score": fake_score}

# ============================================================== #
# TEXT ANALYTICS & FEATURE INSIGHTS
# ============================================================== #
def extract_text_stats(text: str) -> dict:
    """Compute basic statistics about the text for reporting."""
    sentences = re.split(r'[.!?]', text)
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    sent_count = len([s for s in sentences if len(s.strip()) > 0])
    avg_sentence_len = round(word_count / max(sent_count, 1), 2)
    return {
        "word_count": word_count,
        "sentence_count": sent_count,
        "avg_sentence_len": avg_sentence_len
    }

def get_top_keywords(vectorizer, coef_vector, text, top_n=10):
    """Get top influential words for explainability."""
    if coef_vector is None or not hasattr(vectorizer, "vocabulary_"):
        return []
    tfidf = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    scores = (tfidf.toarray()[0] * coef_vector)
    ranked = sorted(
        [(feature_names[i], scores[i]) for i in range(len(feature_names)) if scores[i] != 0],
        key=lambda x: abs(x[1]),
        reverse=True
    )
    return [w for w, _ in ranked[:top_n]]

# ============================================================== #
# TRUSTABILITY
# ============================================================== #
@lru_cache(maxsize=400)
def compute_trustability(url: str) -> dict:
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted_my = [
        "thestar.com.my","malaymail.com","bernama.com","astroawani.com",
        "freemalaysiatoday.com","theedgemalaysia.com","theborneopost.com","themalaysianreserve.com"
    ]
    trusted_global = ["bbc.com","reuters.com","cnn.com","nytimes.com","bloomberg.com","apnews.com"]
    suspicious = ["clickbait","rumor","wordpress","blogspot","infowars.com","breitbart.com"]

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
# EXPLAINABILITY ENGINE
# ============================================================== #
FAKE_KEYWORDS = {"shocking","exclusive","miracle","hoax","exposed","click here","secret","scam","rumor"}
REAL_KEYWORDS = {"official","research","confirmed","report","sources","data","analysis","statement","evidence"}

def _kw_hits(line: str):
    l = line.lower()
    return [w for w in FAKE_KEYWORDS if w in l], [w for w in REAL_KEYWORDS if w in l]

def _tfidf_line_score(line: str, vectorizer, coef_vector):
    if coef_vector is None or not hasattr(vectorizer, "vocabulary_"):
        return 0.0
    score = 0.0
    for tok in tokenize_text(line).split():
        idx = vectorizer.vocabulary_.get(tok)
        if idx is not None and idx < len(coef_vector):
            score += float(coef_vector[idx])
    return score

def _line_heuristics(line: str):
    s = TextBlob(line).sentiment.polarity
    excls = line.count("!")
    words = line.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    fake_pressure = 0.5 * (1 - abs(s)) + 0.3 * upper_ratio + 0.2 * min(excls / 3, 1)
    return 0.5 - fake_pressure

def explain_text(text: str, trust: dict, final_pred: str, model_used: str):
    """
    Generate explainable highlights and reasoning for the analyzed article.
    Each line is evaluated using TF-IDF weights, heuristics, and keyword signals.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    highlighted, reasons = [], []

    # Select appropriate vectorizer and coefficients
    vec, coef = (
        (_my_vectorizer, _my_coef)
        if model_used.lower() == "malay" and _my_vectorizer
        else (_en_vectorizer, _en_coef)
    )

    # Add trust context
    reasons.append(f"Domain '{trust.get('domain')}' categorized as {trust.get('category')}.")

    # Process each line
    for ln in lines:
        w_tfidf = _tfidf_line_score(ln, vec, coef)
        w_heur = _line_heuristics(ln)
        f_hits, r_hits = _kw_hits(ln)

        # Combine keyword signals
        kw_signal = (-0.3 * len(f_hits)) + (0.2 * len(r_hits))
        total = w_tfidf + w_heur + kw_signal

        # Collect human-readable signal tags
        tags = []
        if f_hits:
            tags.append(f"Fake cues: {', '.join(f_hits)}")
        if r_hits:
            tags.append(f"Real cues: {', '.join(r_hits)}")
        if abs(w_tfidf) > 0.2:
            tags.append("Model-weighted term influence")
        if abs(w_heur) > 0.3:
            tags.append("Heuristic tone indicator")

        highlighted.append({
            "text": ln,
            "weight": total,
            "tags": tags
        })

    # Sort lines by influence magnitude (most relevant first)
    highlighted.sort(key=lambda d: abs(d["weight"]), reverse=True)

    # Add high-level reasoning summary
    if final_pred == "Fake":
        reasons.append("Model detected sensational or biased tone in text.")
    elif final_pred == "Likely Real":
        reasons.append("Article appears mostly factual but contains mild bias indicators.")
    else:
        reasons.append("Model detected balanced and factually consistent language.")

    return highlighted[:50], reasons

def adjust_confidence(confidence: float, word_count: int) -> float:
    """
    Smooths out confidence scores:
    - Reduces inflated confidence for short texts
    - Slightly boosts confidence for long, consistent articles
    """
    if word_count < 150:
        confidence *= 0.85
    elif word_count > 1500:
        confidence *= 1.05
    return round(min(confidence, 0.99), 3)



# ============================================================== #
# AUTH HELPERS
# ============================================================== #
def verify_jwt(token: str):
    try:
        return jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"]).get("username")
    except Exception:
        return None

def safe_json(data):
    return jsonify(json.loads(json.dumps(data, default=str)))

# ============================================================== #
# ROUTES
# ============================================================== #
@app.route("/health")
def health():
    return "OK", 200

@app.route("/")
def home():
    return jsonify({"message": "Fake News Detector 101 API ✅"})

# --- AUTH ---
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username, password = data.get("username", "").strip(), data.get("password", "").strip()
    if not username or not password:
        return jsonify({"error": "Missing fields"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    hashed = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()
        return jsonify({"message": "User registered successfully."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username already exists."}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username, password = data.get("username"), data.get("password")
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        row = cur.fetchone()
    if not row or not check_password_hash(row[0], password):
        return jsonify({"error": "Invalid credentials"}), 401
    token = jwt.encode({"username": username, "exp": datetime.utcnow() + timedelta(hours=3)},
                       app.config["JWT_SECRET"], algorithm="HS256")
    return jsonify({"token": token, "username": username})

# --- PREDICT (ENHANCED VERSION) ---
@app.route("/predict", methods=["POST"])
@limiter.limit(PREDICT_LIMIT)
def predict():
    try:
        # ==============================================================
        # 🔑 AUTH CHECK
        # ==============================================================
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)

        # ==============================================================
        # 📥 INPUT VALIDATION
        # ==============================================================
        data = request.get_json() or {}
        text = data.get("text", "").strip()
        headline = data.get("headline", "").strip()
        url = data.get("url", "").strip()

        if not text:
            return jsonify({"error": "Missing text"}), 400  # ✅ fixed indentation

        start_time = time.time()  # ✅ moved correctly inside the function

        # ==============================================================
        # 🧠 MODEL PREDICTION
        # ==============================================================
        ml, _ = predict_fake_news(text)
        ml = dict(ml)
        base_label = "Fake" if ml["prediction"] == "0" else "Real"

        # ==============================================================
        # ⚙️ HEURISTICS + TRUST
        # ==============================================================
        heur = analyze_text_heuristics(text)
        trust = compute_trustability(url)

        # ==============================================================
        # 📊 ADDITIONAL TEXT ANALYSIS
        # ==============================================================
        sentences = re.split(r"[.!?]", text)
        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_len = round(word_count / max(sentence_count, 1), 2)

        article_stats = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_len": avg_sentence_len,
        }

        sentiment_label = (
            "Positive" if heur["sentiment"] > 0.25 else
            "Negative" if heur["sentiment"] < -0.25 else
            "Neutral"
        )

        # ==============================================================
        # 🎚️ MALAY TRUST CORRECTION
        # ==============================================================
        corrected_conf = adjust_confidence(ml["confidence"], article_stats["word_count"])
        final_label = base_label
        if trust["category"].startswith("Trusted (Malaysia)") and base_label == "Fake":
            corrected_conf *= 0.6
            final_label = "Likely Real"

        # ==============================================================
        # 🧩 EXPLAINABILITY
        # ==============================================================
        lines, reasons = explain_text(text, trust, final_label, ml["model_used"])

        # ==============================================================
        # 🔍 TOP KEYWORDS (TF-IDF Influence)
        # ==============================================================
        vec = _my_vectorizer if ml["model_used"].lower() == "malay" else _en_vectorizer
        coef = _my_coef if ml["model_used"].lower() == "malay" else _en_coef

        top_keywords = []
        if coef is not None and hasattr(vec, "vocabulary_"):
            try:
                tfidf = vec.transform([text])
                feature_names = vec.get_feature_names_out()
                weights = (tfidf.toarray()[0] * coef)
                ranked = sorted(
                    [(feature_names[i], weights[i]) for i in range(len(feature_names)) if weights[i] != 0],
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                top_keywords = [w for w, _ in ranked[:10]]
            except Exception:
                top_keywords = []

        # ==============================================================
        # ⚖️ RISK LEVEL
        # ==============================================================
        fake_score = heur.get("fake_score", 0)
        if final_label == "Fake":
            risk_level = "High" if corrected_conf > 0.75 or fake_score > 0.5 else "Medium"
        elif final_label == "Likely Real":
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # ==============================================================
        # 💾 SAVE HISTORY
        # ==============================================================
        if username:
            with get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO scans (username, headline, url, text, prediction, confidence,
                                       heuristics, trustability, language, risk_level, runtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    headline,
                    url,
                    text,
                    final_label,
                    corrected_conf,
                    json.dumps(heur),
                    json.dumps(trust),
                    ml["language"],
                    risk_level,
                    round(time.time() - start_time, 2)
                ))
                conn.commit()

        # ==============================================================
        # 📤 RETURN JSON RESPONSE
        # ==============================================================
        return safe_json({
            "version": "2025.11-ENHANCED",
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "language": ml["language"],
            "model_used": ml["model_used"],
            "prediction": final_label,
            "confidence": corrected_conf,
            "raw_prediction": base_label,
            "risk_level": risk_level,
            "sentiment_label": sentiment_label,
            "article_stats": article_stats,
            "top_keywords": top_keywords,
            "class_probs": ml["class_probs"],
            "heuristics": heur,
            "trustability": trust,
            "explain": {"lines": lines, "reasons": reasons}
        })

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

# --- HISTORY (Consistent with Frontend & Popup) ---
@app.route("/get-history")
def get_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, headline, url, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE username=? ORDER BY timestamp DESC
        """, (username,))
        rows = cur.fetchall()

    history = []
    for r in rows:
        raw_pred = str(r[3]).strip().lower()
        # ✅ Normalize prediction text from database
        if raw_pred in ("0", "fake"):
            pred_label = "Fake"
        elif "likely" in raw_pred:
            pred_label = "Likely Real"
        elif "real" in raw_pred:
            pred_label = "Real"
        else:
            pred_label = "Uncertain"

        history.append({
            "id": r[0],
            "headline": r[1] or "—",
            "url": r[2] or "—",
            "prediction": pred_label,     # ✅ Correct, readable prediction
            "confidence": float(r[4]) if r[4] is not None else None,
            "heuristics": json.loads(r[5]) if r[5] else {},
            "trustability": json.loads(r[6]) if r[6] else {},
            "timestamp": r[7]
        })

    return jsonify({"history": history})

# --- REPORT (Enhanced) ---
@app.route("/get-report/<int:scan_id>")
def get_report(scan_id):
    token = request.args.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return "<h3>Unauthorized</h3>", 401

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, headline, url, text, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE id=? AND username=?
        """, (scan_id, username))
        row = cur.fetchone()

    if not row:
        return "<h3>Report not found.</h3>", 404

    # ==============================================================
    # 📊 Load stored data
    # ==============================================================
    text = row[3] or ""
    heur = json.loads(row[6]) if row[6] else {}
    trust = json.loads(row[7]) if row[7] else {}
    label = str(row[4]).strip()

    # ==============================================================
    # 🌐 Language & Model
    # ==============================================================
    lang = detect_lang(text)
    load_models_if_needed("ms" if lang == "ms" else "en")
    model_used = "Malay" if (lang == "ms" and _my_model and _my_vectorizer) else "English"
    language = "Malay" if lang == "ms" else "English"

    # ==============================================================
    # 📈 Text Statistics
    # ==============================================================
    sentences = re.split(r"[.!?]", text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_sentence_len = round(word_count / max(sentence_count, 1), 2)

    # ✅ Estimate chunks analyzed (assuming ~1500 words per chunk)
    chunks_analyzed = max(1, math.ceil(word_count / 1500))

    article_stats = {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_len": avg_sentence_len,
        "chunks_analyzed": chunks_analyzed
    }

    # ==============================================================
    # 😊 Sentiment Analysis
    # ==============================================================
    sentiment_score = heur.get("sentiment", 0)
    sentiment_label = (
        "Positive" if sentiment_score > 0.25 else
        "Negative" if sentiment_score < -0.25 else
        "Neutral"
    )

    # ==============================================================
    # ⚠️ Risk Level
    # ==============================================================
    fake_score = heur.get("fake_score", 0)
    confidence = float(row[5]) if row[5] else 0
    if "Fake" in label:
        risk_level = "High" if confidence > 0.75 or fake_score > 0.5 else "Medium"
    elif "Likely" in label:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    # ==============================================================
    # 🧠 Top Keywords (TF-IDF Influence)
    # ==============================================================
    vec = _my_vectorizer if language == "Malay" else _en_vectorizer
    coef = _my_coef if language == "Malay" else _en_coef
    top_keywords = []
    if coef is not None and hasattr(vec, "vocabulary_"):
        try:
            tfidf = vec.transform([text])
            feature_names = vec.get_feature_names_out()
            weights = (tfidf.toarray()[0] * coef)
            ranked = sorted(
                [(feature_names[i], weights[i]) for i in range(len(feature_names)) if weights[i] != 0],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_keywords = [w for w, _ in ranked[:10]]
        except Exception as e:
            logging.warning(f"⚠️ Keyword extraction failed: {e}")

    # ==============================================================
    # 🧩 Explainability Highlights
    # ==============================================================
    highlighted_lines, explain_reasons = explain_text(text, trust, label, model_used)

    # ==============================================================
    # ✅ Render full-report.html with rich data
    # ==============================================================
    return render_template(
        "full-report.html",
        row=row,
        heur=heur,
        trust=trust,
        username=username,
        highlighted_lines=highlighted_lines,
        explain_reasons=explain_reasons,
        language=language,
        model_used=model_used,
        sentiment_label=sentiment_label,
        risk_level=risk_level,
        article_stats=article_stats,
        top_keywords=top_keywords,
        chunks_analyzed=chunks_analyzed  # ✅ Correct, self-contained computation
    )

# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Running locally on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
