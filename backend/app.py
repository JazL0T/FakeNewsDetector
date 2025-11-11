# ==============================================================
#  Fake News Detector 101 — Optimized Explainable AI API
#  Version: Render-Ready (2025.11 FINAL - FIXED)
#  Features: Dual EN/MY Models (Malay text only) + Auth + History + Explainability
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, time, threading, hashlib
from datetime import datetime, timedelta
import jwt
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv
import tldextract
from functools import lru_cache
from flask_limiter import Limiter
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
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def get_db_connection():
    conn = sqlite3.connect(app.config["DB_PATH"], timeout=10, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout=10000;")
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
    if not text.strip():
        return {"error": "Empty text"}
    lang = detect_lang(text)
    text_hash = hashlib.sha256(tokenize_text(text).encode()).hexdigest()
    res = _cached_predict(lang, text_hash, text)
    res["language"] = "Malay" if lang == "ms" else "English"
    logging.info(f"🧠 Detected language from text → {res['language']}")
    return res, (lang == "ms" and _my_model and _my_vectorizer)

def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * upper_ratio + 0.3 * min(exclamations / 5, 1)
    return {"sentiment": sentiment, "uppercase_ratio": upper_ratio, "exclamations": exclamations, "fake_score": fake_score}

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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    highlighted, reasons = [], []
    vec, coef = (_my_vectorizer, _my_coef) if model_used.lower() == "malay" and _my_vectorizer else (_en_vectorizer, _en_coef)
    reasons.append(f"Domain '{trust.get('domain')}' categorized as {trust.get('category')}.")

    for ln in lines:
        w_tfidf = _tfidf_line_score(ln, vec, coef)
        w_heur = _line_heuristics(ln)
        f_hits, r_hits = _kw_hits(ln)
        kw_signal = (-0.3 * len(f_hits)) + (0.2 * len(r_hits))
        total = w_tfidf + w_heur + kw_signal
        tags = []
        if f_hits: tags.append(f"Fake cues: {', '.join(f_hits)}")
        if r_hits: tags.append(f"Real cues: {', '.join(r_hits)}")
        if abs(w_tfidf) > 0.2: tags.append("Model-weighted term influence")
        highlighted.append({"text": ln, "weight": total, "tags": tags})

    highlighted.sort(key=lambda d: abs(d["weight"]), reverse=True)
    reasons.append("Model detected sensational/biased tone." if final_pred == "Fake" else "Model detected balanced/factual language.")
    return highlighted[:50], reasons

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

# --- PREDICT (CONSISTENT VERSION) ---
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
            return jsonify({"error": "Missing text"}), 400

        # ==============================================================
        # 🧠 MODEL PREDICTION
        # ==============================================================
        ml, _ = predict_fake_news(text)
        ml = dict(ml)  # make a safe copy (avoid mutation)
        base_label = "Fake" if ml["prediction"] == "0" else "Real"

        # ==============================================================
        # ⚙️ HEURISTICS + TRUST
        # ==============================================================
        heur = analyze_text_heuristics(text)
        trust = compute_trustability(url)

        # ==============================================================
        # 🎚️ MALAY TRUST CORRECTION
        # ==============================================================
        corrected_conf = ml["confidence"]
        final_label = base_label

        if trust["category"].startswith("Trusted (Malaysia)") and base_label == "Fake":
            corrected_conf *= 0.6
            final_label = "Likely Real"

        # ==============================================================
        # 🧩 EXPLAINABILITY
        # ==============================================================
        lines, reasons = explain_text(text, trust, final_label, ml["model_used"])

        # ==============================================================
        # 💾 SAVE HISTORY (store corrected values)
        # ==============================================================
        if username:
            with get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    username,
                    headline,
                    url,
                    text,
                    final_label,          # ✅ Store corrected label
                    corrected_conf,       # ✅ Store corrected confidence
                    json.dumps(heur),
                    json.dumps(trust)
                ))
                conn.commit()

        # ==============================================================
        # 📤 RETURN JSON RESPONSE
        # ==============================================================
        return safe_json({
            "version": "2025.11-FINAL",
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "language": ml["language"],
            "model_used": ml["model_used"],
            "prediction": final_label,           # ✅ Consistent label
            "confidence": corrected_conf,        # ✅ Consistent confidence
            "raw_prediction": base_label,        # 🧠 Optional (for debugging)
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

# --- REPORT ---
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

    heur = json.loads(row[6])
    trust = json.loads(row[7])

    # ✅ Use stored prediction directly (no recomputation)
    label = str(row[4]).strip()

    # ✅ Detect language for highlighting only
    text = row[3] or ""
    lang = detect_lang(text)
    load_models_if_needed("ms" if lang == "ms" else "en")

    # ✅ Choose correct model/language names
    model_used = "Malay" if (lang == "ms" and _my_model and _my_vectorizer) else "English"
    language = "Malay" if lang == "ms" else "English"

    # ✅ Generate explainable highlights using stored label
    highlighted_lines, explain_reasons = explain_text(text, trust, label, model_used)

    # ✅ Pass everything to full-report.html
    return render_template(
        "full-report.html",
        row=row,
        heur=heur,
        trust=trust,
        username=username,
        highlighted_lines=highlighted_lines,
        explain_reasons=explain_reasons,
        language=language,
        model_used=model_used
    )

# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Running locally on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
