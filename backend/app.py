# ==============================================================
#  Fake News Detector API (Explainable) — External Template + Auth Fix
# ==============================================================

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib, re, os, sqlite3, logging, json, math
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
    # Try to expose linear model weights if present (e.g., LogisticRegression / LinearSVC)
    if hasattr(model, "coef_"):
        coef_arr = getattr(model, "coef_", None)
        if coef_arr is not None:
            # Binary classifier -> shape (1, n_features)
            coef_vector = coef_arr[0]
            # get_feature_names_out is available in newer scikit versions
            if hasattr(vectorizer, "get_feature_names_out"):
                vocab = vectorizer.get_feature_names_out()
            elif hasattr(vectorizer, "vocabulary_"):
                # Fallback: build index->term mapping from vocabulary_
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
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        """
        )
        c.execute(
            """
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
        """
        )
        conn.commit()


def get_db_connection():
    return sqlite3.connect(app.config["DB_PATH"])


init_db()

# ============================================================== #
# HELPER: Tokenization / Prediction / Heuristics
# ============================================================== #
def tokenize_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


def predict_fake_news(text: str) -> dict:
    if not model or not vectorizer:
        return {"error": "Model not loaded"}
    features = vectorizer.transform([tokenize_text(text)])
    pred = model.predict(features)[0]
    # Try probabilities if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        conf = float(max(probs))
        class_probs = {"0": float(probs[0]), "1": float(probs[1])}
    else:
        # Probability not available — synthesize confidence from decision function if present
        if hasattr(model, "decision_function"):
            df = model.decision_function(features)
            # squash with logistic for pseudo-confidence
            conf = float(1 / (1 + math.exp(-abs(df[0]))))
        else:
            conf = 0.5
        class_probs = {"0": 1 - conf, "1": conf}
    return {
        "prediction": str(pred),
        "confidence": conf,
        "class_probs": class_probs,
    }


def analyze_text_heuristics(text: str) -> dict:
    blob = TextBlob(text)
    sentiment = float(blob.sentiment.polarity)
    words = text.split()
    uppercase_ratio = sum(1 for w in words if w.isupper()) / max(1, len(words))
    exclamations = text.count("!")
    fake_score = (
        0.4 * (1 - abs(sentiment)) + 0.3 * uppercase_ratio + 0.3 * min(exclamations / 5, 1)
    )
    return {
        "sentiment": sentiment,
        "uppercase_ratio": uppercase_ratio,
        "exclamations": exclamations,
        "fake_score": fake_score,
    }


def compute_trustability(url: str) -> dict:
    domain = tldextract.extract(url).registered_domain or "unknown"
    trusted_sources = [
        "bbc.com",
        "reuters.com",
        "apnews.com",
        "nytimes.com",
        "theguardian.com",
        "npr.org",
    ]
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
# EXPLAINABILITY UTILITIES
# ============================================================== #

FAKE_KEYWORDS = {
    "shocking", "exclusive", "miracle", "hoax", "exposed", "coverup", "you wont believe",
    "click here", "secret", "viral", "fake", "rumor", "scam"
}
REAL_KEYWORDS = {
    "official", "research", "confirmed", "report", "statement", "evidence", "investigation",
    "sources", "analysis", "according", "data"
}

def keyword_hits(line: str):
    l = line.lower()
    f = [w for w in FAKE_KEYWORDS if w in l]
    r = [w for w in REAL_KEYWORDS if w in l]
    return f, r

def tfidf_line_score(line: str):
    """
    Sum model weights for tokens in the line.
    Positive -> pushes toward class '1' (we map to 'Real'),
    Negative -> toward class '0' ('Fake').
    Returns score (float). If no coef available, returns 0.
    """
    if coef_vector is None or vectorizer is None:
        return 0.0
    if hasattr(vectorizer, "build_analyzer"):
        analyzer = vectorizer.build_analyzer()
        tokens = analyzer(line)
    else:
        tokens = tokenize_text(line).split()
    score = 0.0
    if hasattr(vectorizer, "vocabulary_"):
        vocab_map = vectorizer.vocabulary_
        for t in tokens:
            idx = vocab_map.get(t)
            if idx is not None and idx < len(coef_vector):
                score += float(coef_vector[idx])
    else:
        # no direct vocab map; best effort
        if vocab is not None:
            idx_map = {term: i for i, term in enumerate(vocab)}
            for t in tokens:
                i = idx_map.get(t)
                if i is not None and i < len(coef_vector):
                    score += float(coef_vector[i])
    return score

def line_heuristic_score(line: str):
    # simple heuristic signals for the line itself
    excls = line.count("!")
    upper_ratio = 0.0
    words = line.split()
    if words:
        upper_ratio = sum(1 for w in words if w.isupper()) / len(words)
    s = TextBlob(line).sentiment.polarity
    # Map to a rough “fake pressure” score ( >0 fake-ish, <0 real-ish )
    fake_pressure = (0.5 * (1 - abs(s))) + (0.3 * upper_ratio) + (0.2 * min(excls / 3, 1))
    return fake_pressure  # ~0..1

def explain_text(text: str, trust: dict, final_pred: str):
    """
    Produce:
      - highlighted_lines: list of dicts {text, weight, reason_tags}
      - reasons: list[str]
    Weight sign:
       positive => more REAL (green)
       negative => more FAKE (red)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    highlighted = []
    reasons = []

    # domain reason
    if trust.get("category") == "Trusted":
        reasons.append(f"Domain '{trust.get('domain')}' is in trusted list.")
    elif trust.get("category") == "Suspicious":
        reasons.append(f"Domain '{trust.get('domain')}' has suspicious markers.")
    else:
        reasons.append(f"Domain '{trust.get('domain')}' has uncertain reliability.")

    # per-line scores and tags
    for ln in lines:
        # TF-IDF directional score (pos -> real, neg -> fake)
        w_tfidf = tfidf_line_score(ln)
        # heuristic fake pressure (0..1), map to signed signal (fake-positive => negative weight)
        fp = line_heuristic_score(ln)
        # 0.5 baseline; >0.5 -> fake-ish -> negative, <0.5 -> real-ish -> positive
        w_heur = (0.5 - fp)

        f_hits, r_hits = keyword_hits(ln)
        kw_signal = 0.0
        if f_hits:
            kw_signal -= 0.25 * len(f_hits)  # fake-ish
        if r_hits:
            kw_signal += 0.15 * len(r_hits)  # real-ish

        total = w_tfidf + w_heur + kw_signal

        tags = []
        if f_hits:
            tags.append(f"Fake cues: {', '.join(f_hits)}")
        if r_hits:
            tags.append(f"Real cues: {', '.join(r_hits)}")
        if abs(w_tfidf) > 0.2:
            tags.append("Model-weighted terms influence")
        if fp > 0.7:
            tags.append("Sensational tone (caps/exclamations)")
        elif fp < 0.3:
            tags.append("Balanced tone")

        highlighted.append({"text": ln, "weight": total, "tags": tags})

    # global explanation bullets
    # We only have document-level heuristics in /predict; for /get-report we’ll add reasons here
    if final_pred == "Fake":
        reasons.append("Model detected terms and tone consistent with fake/sensational content.")
    elif final_pred == "Real":
        reasons.append("Model detected balanced language consistent with real reporting.")
    else:
        reasons.append("Signals were mixed; result uncertain.")

    # Sort lines by absolute contribution, descending (for a top-5 UI if needed)
    highlighted_sorted = sorted(highlighted, key=lambda d: abs(d["weight"]), reverse=True)

    return highlighted_sorted, reasons


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

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters long."}), 400

    hashed_pw = generate_password_hash(password)
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_pw)
            )
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


# ---------- PREDICT ----------
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
        return jsonify({"error": ml["error"]}), 500
    heur = analyze_text_heuristics(text)
    trust = compute_trustability(url)

    # explain (safe for popup to ignore)
    final_pred_label = "Fake" if ml["prediction"] == "0" else "Real"
    highlighted_lines, reasons = explain_text(text, trust, final_pred_label)

    if username:
        with get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO scans (username, headline, url, text, prediction, confidence, heuristics, trustability)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    username,
                    headline,
                    url,
                    text,
                    ml["prediction"],
                    ml["confidence"],
                    json.dumps(heur),
                    json.dumps(trust),
                ),
            )
            conn.commit()

    # Keep original fields for popup compatibility; add extra "explain" fields
    return safe_json(
        {
            "username": username or "Guest",
            "headline": headline,
            "url": url,
            "prediction": final_pred_label,
            "confidence": ml["confidence"],
            "class_probs": ml["class_probs"],
            "heuristics": heur,
            "trustability": trust,
            "explain": {
                "lines": highlighted_lines[:50],  # cap for payload safety
                "reasons": reasons,
            },
        }
    )


# ---------- HISTORY ----------
@app.route("/get-history", methods=["GET"])
def get_history():
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    username = verify_jwt(token)
    if not username:
        return jsonify({"error": "Unauthorized"}), 401
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, headline, url, prediction, confidence, heuristics, trustability, timestamp
            FROM scans WHERE username = ? ORDER BY timestamp DESC
        """,
            (username,),
        )
        rows = cur.fetchall()
    history = []
    for r in rows:
        history.append(
            {
                "id": r[0],
                "headline": r[1],
                "url": r[2],
                "prediction": "Fake" if r[3] == "0" else "Real",
                "confidence": r[4],
                "heuristics": json.loads(r[5]),
                "trustability": json.loads(r[6]),
                "timestamp": r[7],
            }
        )
    return jsonify({"history": history})


# ---------- FULL REPORT ----------
@app.route("/get-report/<int:scan_id>")
def get_report(scan_id):
    token = request.args.get("token")
    if not token:
        token = request.headers.get("Authorization", "").replace("Bearer ", "")

    username = verify_jwt(token)
    if not username:
        return "<h3>Unauthorized - Please log in first.</h3>", 401

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
        return "<h3>Report not found or not owned by you.</h3>", 404

    heur = json.loads(row[6])
    trust = json.loads(row[7])

    # Build explainability for the HTML template
    final_pred_label = "Fake" if row[4] == "0" else "Real"
    highlighted_lines, reasons = explain_text(row[3] or "", trust, final_pred_label)

    # The template (templates/full-report.html) can optionally use:
    #   highlighted_lines: [{text, weight, tags}], reasons: [str]
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
