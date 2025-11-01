# ==============================================================
#  Fake News Detector API (Explainable) — Auth-Required Version
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, math
from datetime import datetime, timedelta
import jwt
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from dotenv import load_dotenv
import tldextract

# ============================================================== #
# CONFIGURATION
# ============================================================== #
load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app, origins=["*"])

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

# ============================================================== #
# LOGGING
# ============================================================== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ============================================================== #
# LOAD MODEL
# ============================================================== #
model, vectorizer = None, None
coef_vector = None
vocab = None
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if hasattr(model, "coef_"):
        coef_arr = getattr(model, "coef_", None)
        if coef_arr is not None:
            coef_vector = coef_arr[0]
            if hasattr(vectorizer, "get_feature_names_out"):
                vocab = vectorizer.get_feature_names_out()
            elif hasattr(vectorizer, "vocabulary_"):
                inv = {i: t for t, i in vectorizer.vocabulary_.items()}
                vocab = [inv[i] for i in range(len(inv))]
    logging.info("✅ Model and vectorizer loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model/vectorizer: {e}")

# ============================================================== #
# DATABASE SETUP
# ============================================================== #
def init_db():
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
# AUTH DECORATOR — require_login
# ============================================================== #
def verify_jwt(token: str):
    try:
        decoded = jwt.decode(token, app.config["JWT_SECRET"], algorithms=["HS256"])
        return decoded.get("username")
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def require_login(f):
    """Decorator: only allow if valid JWT present"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        username = verify_jwt(token)
        if not username:
            return jsonify({"error": "Unauthorized. Please log in first."}), 401
        return f(username, *args, **kwargs)
    return wrapper


# ============================================================== #
# HELPER FUNCTIONS
# ============================================================== #
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def safe_json(data: dict):
    return jsonify(json.loads(json.dumps(data, default=str)))


# ============================================================== #
# MODEL & HEURISTICS
# ============================================================== #
def predict_fake_news(text: str) -> dict:
    if not model or not vectorizer:
        return {"error": "Model not loaded"}
    features = vectorizer.transform([tokenize_text(text)])
    pred = model.predict(features)[0]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        conf = float(max(probs))
        class_probs = {"0": float(probs[0]), "1": float(probs[1])}
    else:
        conf = 0.5
        class_probs = {"0": 1 - conf, "1": conf}
    return {"prediction": str(pred), "confidence": conf, "class_probs": class_probs}


def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = float(blob.sentiment.polarity)
    words = text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = 0.4 * (1 - abs(sentiment)) + 0.3 * uppercase_ratio + 0.3 * min(exclamations / 5, 1)
    return {
        "sentiment": sentiment,
        "uppercase_ratio": uppercase_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score,
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


# ============================================================== #
# ROUTES
# ============================================================== #

@app.route("/")
def home():
    return jsonify({"message": "✅ Fake News Detector API is running!"})


# ---------- AUTH ----------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

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

    token = jwt.encode(
        {"username": username, "exp": datetime.utcnow() + timedelta(hours=3)},
        app.config["JWT_SECRET"],
        algorithm="HS256",
    )
    return jsonify({"token": token, "username": username})


# ---------- PREDICT (Auth Required) ----------
@app.route("/predict", methods=["POST"])
@require_login
def predict(username):
    data = request.get_json() or {}
    text = data.get("text", "")
    headline = data.get("headline", "")
    url = data.get("url", "")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    ml = predict_fake_news(text)
    if "error" in ml:
        return jsonify({"error": ml["error"]}), 500

    heur = analyze_text_heuristics(text)
    trust = compute_trustability(url)

    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, headline, url, text, ml["prediction"], ml["confidence"], json.dumps(heur), json.dumps(trust)))
        conn.commit()

    return safe_json({
        "username": username,
        "headline": headline,
        "url": url,
        "prediction": "Fake" if ml["prediction"] == "0" else "Real",
        "confidence": ml["confidence"],
        "heuristics": heur,
        "trustability": trust
    })


# ---------- HISTORY (Auth Required) ----------
@app.route("/get-history", methods=["GET"])
@require_login
def get_history(username):
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
            "timestamp": r[7],
        })
    return jsonify({"history": history})


# ---------- FULL REPORT ----------
@app.route("/get-report/<int:scan_id>")
def get_report(scan_id):
    # ✅ Accept token from query param (preferred for Chrome extension)
    token = request.args.get("token") or request.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not token:
        return jsonify({"error": "Unauthorized. Missing token parameter."}), 401

    username = verify_jwt(token)
    if not username:
        return jsonify({"error": "Unauthorized. Please log in first."}), 401

    # ✅ Fetch scan data owned by the logged-in user
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, headline, url, text, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE id = ? AND username = ?
            """,
            (scan_id, username),
        )
        row = cur.fetchone()

    if not row:
        return jsonify({"error": "Report not found or not owned by this user."}), 404

    # ✅ Load JSON columns
    heur = json.loads(row[6]) if row[6] else {}
    trust = json.loads(row[7]) if row[7] else {}

    # ✅ Determine prediction label
    final_pred_label = "Fake" if row[4] == "0" else "Real"

    # ✅ Generate explainability data
    highlighted_lines, reasons = explain_text(row[3] or "", trust, final_pred_label)

    # ✅ Render the professional HTML report
    return render_template(
        "full-report.html",
        row=row,
        heur=heur,
        trust=trust,
        username=username,
        highlighted_lines=highlighted_lines,
        explain_reasons=reasons,
    )


# ============================================================== #
# MAIN
# ============================================================== #
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    logging.info(f"🚀 Server running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
