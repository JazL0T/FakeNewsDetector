# ==============================================================
#  Fake News Detector 101 — Optimized Explainable AI API
#  Version: Render-Ready (2025.11 FINAL - FIXED)
#  Features: Dual EN/MY Models (Malay text only) + Auth + History + Explainability
# ==============================================================

# ==============================================================
#  Gevent Compatibility Patch (must be FIRST)
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

from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0

import tldextract
# ============================================================== #
# TLDExtract (Offline Mode - Prevent SSL Recursion)
# ============================================================== #
_TLD_EXTRACTOR = tldextract.TLDExtract(
    cache_dir="/tmp/tldextract_cache",  # ✅ Safe writable cache dir on Render
    suffix_list_urls=None,              # ✅ Fully offline (no HTTPS requests)
    fallback_to_snapshot=True           # ✅ Uses built-in snapshot of domain suffixes
)

# ✅ Optional sanity check (helps verify on first deploy)
try:
    test_domain = _TLD_EXTRACTOR("https://www.bbc.com").registered_domain
    logging.info(f"🧩 TLD extractor initialized successfully (test: {test_domain})")
except Exception as e:
    logging.warning(f"⚠️ TLD extractor init failed: {e}")

from functools import lru_cache
from flask_limiter import Limiter
from langdetect import detect, LangDetectException
from logging.handlers import RotatingFileHandler
import threading, requests

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
# AUTO PING
# ============================================================== #

def auto_ping():
    """Keeps the app alive by pinging itself every 5 minutes."""
    while True:
        try:
            url = os.getenv("RENDER_EXTERNAL_URL", "")
            if url:
                requests.get(f"{url}/health", timeout=10)
                logging.info("🌐 Self-ping successful")
        except Exception as e:
            logging.warning(f"⚠️ Self-ping failed: {e}")
        time.sleep(300)  # every 5 minutes

# Start background thread
threading.Thread(target=auto_ping, daemon=True).start()

def log_status():
    """Logs backend stats (users, scans, uptime, and model load) every 10 minutes."""
    app_start = time.time()
    while True:
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM users")
                users = cur.fetchone()[0]
                cur.execute("SELECT COUNT(*) FROM scans")
                scans = cur.fetchone()[0]

            uptime = round(time.time() - app_start, 1)
            status_summary = (
                f"📊 STATUS — Users: {users} | Scans: {scans} | "
                f"EN_Model: {'✅' if _en_model else '❌'} | "
                f"MY_Model: {'✅' if _my_model else '❌'} | "
                f"Uptime: {uptime:.0f}s"
            )
            logging.info(status_summary)

        except Exception as e:
            logging.warning(f"⚠️ Failed to log status: {e}")

        time.sleep(600)  # every 10 minutes

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

        # -------------------------
        # USERS TABLE
        # -------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                is_admin INTEGER DEFAULT 0
            )
        """)

        # -------------------------
        # SCANS TABLE
        # -------------------------
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

        # -------------------------
        # ✅ NEW: LOGS TABLE
        # -------------------------
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                action TEXT,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()

def create_default_admin():
    admin_user = os.getenv("ADMIN_USERNAME")
    admin_pass = os.getenv("ADMIN_PASSWORD")

    if not admin_user or not admin_pass:
        print("❌ ADMIN_USERNAME or ADMIN_PASSWORD not set in ENV!")
        return

    hashed = generate_password_hash(admin_pass)

    with sqlite3.connect(app.config["DB_PATH"]) as conn:
        cur = conn.cursor()

        # Check if admin exists
        cur.execute("SELECT id, username FROM users WHERE is_admin=1")
        row = cur.fetchone()

        if row:
            existing_admin = row[1]

            # If admin username changed → rename user
            if existing_admin != admin_user:
                cur.execute("UPDATE users SET username=? WHERE id=?", (admin_user, row[0]))

            # Always update password for security
            cur.execute("UPDATE users SET password=? WHERE id=?", (hashed, row[0]))

            conn.commit()
            print(f"🔄 Admin updated from ENV: {admin_user}")
            return

        # If no admin exists → create new one
        cur.execute("""
            INSERT INTO users (username, password, is_admin)
            VALUES (?, ?, 1)
        """, (admin_user, hashed))

        conn.commit()
        print(f"✅ Admin created from ENV: {admin_user}")

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
create_default_admin()

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

def predict_fake_news(text: str, url: str = ""):
    """
    Perform AI-based fake news prediction with dual-model (EN/MY) selection.
    - Uses safe_detect_language() for domain-aware detection.
    - Caches short texts and analyzes long articles in chunks.
    """
    if not text.strip():
        return {"error": "Empty text"}

    # --- Smart language detection (aware of domain + content) ---
    lang = safe_detect_language(text, url)
    logging.info(f"[LangDetect] {url} → {lang}")

    # --- Load correct model based on detected language ---
    if lang == "Malay":
        load_models_if_needed("ms")
        model, vec, coef, used = _my_model, _my_vectorizer, _my_coef, "Malay"
    else:
        load_models_if_needed("en")
        model, vec, coef, used = _en_model, _en_vectorizer, _en_coef, "English"

    # --- Prepare text and cache hash ---
    clean_text = tokenize_text(text)
    text_hash = hashlib.sha256(clean_text.encode()).hexdigest()
    word_count = len(clean_text.split())

    # --- For short text (<700 words): use cache for speed ---
    if word_count < 700:
        res = _cached_predict("ms" if used == "Malay" else "en", text_hash, text)
        res["language"] = lang
        res["chunks_analyzed"] = 1
        logging.info(f"🧠 Cached prediction used for short text ({word_count} words) → {used}")
        return res, (used == "Malay")

    # --- For long text: split and analyze in chunks ---
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

    # --- Evaluate each chunk ---
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

    # --- Aggregate results ---
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
        "language": lang,
        "chunks_analyzed": len(chunks),
    }

    logging.info(
        f"🧠 Long article detected ({word_count} words) | "
        f"{len(chunks)} chunks analyzed | Model: {used} | Language: {lang}"
    )

    return result, (used == "Malay")

def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    words = text.split()
    upper_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * upper_ratio + 0.3 * min(exclamations / 5, 1)
    return {"sentiment": sentiment, "uppercase_ratio": upper_ratio, "exclamations": exclamations, "fake_score": fake_score}

def add_log(username, action):
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO logs (username, action) VALUES (?, ?)",
                (username, action)
            )
            conn.commit()
    except Exception as e:
        logging.warning(f"⚠️ Failed to write log: {e}")

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
    """
    Determines the trust category of a given URL domain.
    - Uses offline suffix extraction to avoid SSL recursion errors on Render.
    - Normalizes malformed URLs like 'cnn.com' or 'www.reuters.com'.
    - Returns structured metadata (domain, score, category).
    """

    try:
        # 🧹 Normalize URL before extraction
        if not url or not isinstance(url, str):
            domain = "unknown"
        else:
            clean_url = url.strip()
            # Ensure URL has a valid scheme (Render-safe)
            if not re.match(r"^https?://", clean_url):
                clean_url = "https://" + clean_url
            # ✅ Use preloaded offline extractor (no external requests)
            domain = _TLD_EXTRACTOR(clean_url).registered_domain or "unknown"
    except Exception as e:
        logging.warning(f"⚠️ Domain extraction failed for '{url}': {e}")
        domain = "unknown"

    # --- TRUSTED SOURCES ---
    trusted_my = [

        "thestar.com.my", "malaymail.com", "bernama.com", "astroawani.com", "freemalaysiatoday.com",
        "theedgemalaysia.com", "theborneopost.com", "themalaysianreserve.com", "nst.com.my",
        "utusan.com.my", "malaysiakini.com", "dailyexpress.com.my", "sinarharian.com.my", "kosmo.com.my",
        "bharian.com.my", "hmetro.com.my"
    ]

    trusted_global = [
        # 🌍 Major international outlets
        "bbc.com", "reuters.com", "cnn.com", "nytimes.com", "bloomberg.com",
        "apnews.com", "theguardian.com", "washingtonpost.com", "npr.org",
        "aljazeera.com", "cnbc.com", "forbes.com", "time.com", "dw.com",
        "economist.com", "abcnews.go.com", "usatoday.com", "sky.com", "pbs.org",
        "politico.com", "financialtimes.com", "thehill.com", "vox.com", "boston.com"
    ]

    fact_checkers = [
        # ✅ Verified fact-checking organizations
        "politifact.com", "snopes.com", "factcheck.org", "afp.com", "afpfactcheck.com",
        "boomlive.in", "malaysiakini.com/factcheck", "thescoop.co", "factly.in",
        "reuters.com/fact-check", "apnews.com/fact-check"
    ]

    suspicious = [
        # ⚠️ Common misinformation / low-credibility domains
        "clickbait", "rumor", "wordpress", "blogspot", "medium.com", "substack.com",
        "infowars.com", "breitbart.com", "naturalnews.com", "thegatewaypundit.com",
        "sputniknews.com", "rt.com", "zerohedge.com", "newsmax.com", "oan.com",
        "beforeitsnews.com", "dailyexpose.uk", "worldtruth.tv", "newspunch.com",
        "yournewswire.com", "patriotpost.us", "theblaze.com", "rumble.com",
        "bitchute.com", "truthsocial.com", "gab.com", "duckduckgo.com/news"
    ]

    # --- CLASSIFY DOMAIN ---
    if any(fc in domain for fc in fact_checkers):
        return {"domain": domain, "trust_score": 95, "category": "Verified Fact-Checker"}
    if any(t in domain for t in trusted_my):
        return {"domain": domain, "trust_score": 90, "category": "Trusted (Malaysia)"}
    if any(t in domain for t in trusted_global):
        return {"domain": domain, "trust_score": 90, "category": "Trusted"}
    if any(s in domain for s in suspicious):
        return {"domain": domain, "trust_score": 30, "category": "Suspicious"}
    if ".my" in domain:
        return {"domain": domain, "trust_score": 60, "category": "Unverified Malaysian Source"}

    # --- Default fallback ---
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

def safe_detect_language(text, url=""):
    """
    Detects language for the article and ensures correct model mapping.
    - Malay text or Indonesian language → 'Malay'
    - English text → 'English'
    - Mixed or uncertain → fallback to 'English'
    """
    try:
        langs = detect_langs(text)
        top = langs[0]
        lang = top.lang

        # --- Detect true text language ---
        if lang in ["ms", "id"]:
            detected_lang = "Malay"
        elif lang == "en":
            detected_lang = "English"
        else:
            detected_lang = "English"  # default fallback

        # --- Override based on domain for Malaysian sites ---
        malaysian_domains = [
            "astroawani", "bernama", "freemalaysiatoday",
            "malaymail", "thestar", "nst", "utusan", "theborneopost"
        ]

        # --- Check if the URL belongs to a Malaysian outlet ---
        if any(site in url.lower() for site in malaysian_domains):
            # ✅ If it's a Malaysian site but text looks English, keep English model
            # ✅ If it's Malay text, use Malay model
            if detected_lang == "Malay":
                return "Malay"
            else:
                return "English"

        # --- Default for other international or unknown sources ---
        return detected_lang

    except Exception as e:
        print(f"[LangDetect Error] {e}")
        return "English"

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

def verify_admin(token):
    try:
        payload = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return payload.get("is_admin", False)
    except:
        return False

def require_admin():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    try:
        data = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        if not data.get("is_admin"):
            return None, jsonify({"error": "Admin access required"}), 403
        return data.get("username"), None, None
    except Exception:
        return None, jsonify({"error": "Invalid or missing token"}), 401

# ============================================================== #
# ROUTES
# ============================================================== #
@app.route("/health")
def health():
    utc_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    logging.info(f"🩺 Health check at {utc_now}")
    return jsonify({
        "status": "ok",
        "timestamp": utc_now,
        "version": "2025.11-FINAL"
    }), 200

@app.route("/status")
def status():
    """Returns live API health and usage statistics."""
    try:
        # --- Collect counts from the database ---
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM users")
            users = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM scans")
            scans = cur.fetchone()[0]

        # --- Calculate uptime (since app start) ---
        uptime_seconds = round(time.time() - psutil.boot_time(), 1) if hasattr(__import__('psutil'), 'boot_time') else None

        # --- Prepare the response ---
        return jsonify({
            "status": "ok",
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "users_registered": users,
            "total_scans": scans,
            "models_loaded": {
                "english_model": bool(_en_model),
                "malay_model": bool(_my_model)
            },
            "rate_limit": PREDICT_LIMIT,
            "uptime_seconds": uptime_seconds,
            "message": "✅ API is active and healthy"
        }), 200

    except Exception as e:
        logging.exception("⚠️ Failed to retrieve status")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

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

    try:
        hashed = generate_password_hash(password)
        with get_db_connection() as conn:
            conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
            conn.commit()

        add_log(username, "Registered new account")
        return jsonify({"message": "User registered successfully."})

    except sqlite3.IntegrityError:
        add_log(username, "Failed registration (username exists)")
        return jsonify({"error": "Username already exists."}), 400

@app.route("/login", methods=["POST"])
@limiter.limit("5 per minute")   # 🛡️ Brute-force protection
def login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password, is_admin FROM users WHERE username=?", (username,))
        row = cur.fetchone()

    if not row or not check_password_hash(row[0], password):
        add_log(username or "Unknown", "Failed login attempt")
        return jsonify({"error": "Invalid credentials"}), 401

    hashed_password, is_admin = row
    is_admin = bool(is_admin)

    # 🔐 Create JWT with role included
    token = jwt.encode({
        "username": username,
        "is_admin": is_admin,
        "exp": datetime.utcnow() + timedelta(hours=3)
    }, app.config["JWT_SECRET"], algorithm="HS256")

    # ✅ Log successful login
    add_log(username, "Logged in successfully")

    # Frontend-safe response
    return jsonify({
        "token": token,
        "username": username,
        "is_admin": is_admin
    })

@app.route("/admin/login", methods=["POST"])
@limiter.limit("5 per minute")
def admin_login():
    data = request.get_json() or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password, is_admin FROM users WHERE username=?", (username,))
        row = cur.fetchone()

    if not row:
        add_log(username, "Failed admin login — user not found")
        return jsonify({"error": "Invalid admin credentials"}), 401

    hashed_pw, is_admin = row
    is_admin = bool(is_admin)

    if not is_admin:
        add_log(username, "Failed admin login — not admin")
        return jsonify({"error": "Unauthorized"}), 403

    if not check_password_hash(hashed_pw, password):
        add_log(username, "Failed admin login — wrong password")
        return jsonify({"error": "Invalid admin credentials"}), 401

    # Create admin token
    token = jwt.encode({
        "username": username,
        "is_admin": True,
        "exp": datetime.utcnow() + timedelta(hours=3)
    }, app.config["JWT_SECRET"], algorithm="HS256")

    add_log(username, "Admin login success")

    return jsonify({
        "token": token,
        "username": username,
        "is_admin": True
    })

# -----------------------
# Admin endpoints (fixed)
# -----------------------

@app.route("/admin/users", methods=["GET"])
def admin_list_users():

    # 🔐 Layer 2: JWT admin check
    admin, error, code = require_admin()
    if error:
        add_log("Unknown", "Unauthorized attempt to access user list")
        return error, code

    # 📝 Log admin action
    add_log(admin, "Viewed user list")

    # 📥 Load all users
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, is_admin FROM users ORDER BY id DESC")
        rows = cur.fetchall()

    users = [{"id": r[0], "username": r[1], "is_admin": bool(r[2])} for r in rows]

    return jsonify({"admin": admin, "users": users}), 200

@app.route("/admin/delete-user/<string:username>", methods=["DELETE"])
def admin_delete_user(username):

    # 🔐 Layer 2: JWT admin validation
    admin, error, code = require_admin()
    if error:
        add_log("Unknown", f"Unauthorized attempt to delete user: {username}")
        return error, code

    # ❌ Prevent deleting own admin account
    if admin == username:
        add_log(admin, f"Attempted to delete own admin account (blocked)")
        return jsonify({"error": "Cannot delete your own admin account"}), 403

    with get_db_connection() as conn:
        cur = conn.cursor()

        # 🔍 Check if target user exists
        cur.execute("SELECT id, is_admin FROM users WHERE username=?", (username,))
        row = cur.fetchone()

        if not row:
            add_log(admin, f"Tried to delete non-existing user: {username}")
            return jsonify({"error": "User not found"}), 404

        # ⛔ Prevent deleting admin accounts
        if int(row["is_admin"]) == 1:
            add_log(admin, f"Tried to delete admin account: {username} (blocked)")
            return jsonify({"error": "Cannot delete admin accounts"}), 403

        # 🗑️ Proceed with deletion
        cur.execute("DELETE FROM scans WHERE username=?", (username,))
        cur.execute("DELETE FROM users WHERE username=?", (username,))
        conn.commit()

    # 📝 Log successful deletion
    add_log(admin, f"Deleted user: {username}")

    return jsonify({
        "message": f"User '{username}' and related scans deleted",
        "performed_by": admin
    }), 200

@app.route("/admin/stats", methods=["GET"])
def admin_stats():

    admin, error, code = require_admin()
    if error:
        return error, code

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM users")
        users = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM scans")
        scans = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM scans WHERE prediction LIKE 'Fake%'")
        fake_count = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM scans WHERE prediction LIKE 'Real%'")
        real_count = cur.fetchone()["cnt"]

    return jsonify({
        "admin": admin,
        "stats": {
            "total_users": users,
            "total_scans": scans,
            "fake_scans": fake_count,
            "real_scans": real_count,
            "malay_model_loaded": bool(_my_model),
            "english_model_loaded": bool(_en_model)
        }
    }), 200

@app.route("/admin/logs", methods=["GET"])
def admin_logs():

    admin, error, code = require_admin()
    if error:
        return error, code

    allowed = ("login", "logout", "register", "scan")  # <-- only these allowed

    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT username, action, time
            FROM logs
            WHERE LOWER(action) LIKE '%login%'
               OR LOWER(action) LIKE '%logout%'
               OR LOWER(action) LIKE '%register%'
               OR LOWER(action) LIKE '%scan%'
            ORDER BY time DESC
            LIMIT 200
        """)
        rows = cur.fetchall()

    logs = [
        {
            "username": row["username"],
            "action": row["action"],
            "time": row["time"],
        }
        for row in rows
    ]

    return jsonify({"logs": logs}), 200

@app.route("/log-logout", methods=["POST"])
def log_logout():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if username:
        add_log(username, "Logged out")
    return jsonify({"status": "ok"})

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
            return jsonify({"error": "Missing text"}), 400

        start_time = time.time()

        # ==============================================================
        # 🧠 MODEL PREDICTION
        # ==============================================================
        ml, _ = predict_fake_news(text, url)
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
        # 🎚️ TRUST CORRECTION
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
        # 🔍 TOP KEYWORDS
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
        # 💾 SAVE HISTORY + LOG SCAN
        # ==============================================================
        try:
            if username:
                add_log(username, "Performed scan")

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
            else:
                add_log("Guest", "Performed scan")
        except Exception as e:
            logging.warning(f"⚠️ Failed to save scan/log: {e}")

        # ==============================================================
        # 📤 RETURN RESPONSE
        # ==============================================================
        return safe_json({
            "version": "2025.11-ENHANCED",
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "language": safe_detect_language(text, url),
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

    # 🌐 Language & Model (FIXED)
    language = safe_detect_language(text, row[2] or "")
    model_used = "Malay" if language == "Malay" else "English"
    load_models_if_needed("ms" if language == "Malay" else "en")

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
    port = int(os.environ.get("PORT", 10000))  # ✅ Render sets this automatically
    logging.info(f"🚀 Starting server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
