# backend/app.py
import os
import time
import logging
import re
import json
import pickle
from urllib.parse import urlparse
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, render_template_string, abort
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import jwt
from textblob import TextBlob

# ---------------- Config ----------------
APP_NAME = "FakeNewsDetector"
PORT = int(os.environ.get("PORT", 10000))
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH", "vectorizer.pkl")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data.db")
JWT_SECRET = os.environ.get("JWT_SECRET", os.environ.get("SECRET_KEY", "please_change_this"))
JWT_ALGO = "HS256"
JWT_EXP_DAYS = int(os.environ.get("JWT_EXP_DAYS", 7))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")  # set to your domain/extension in production

# ---------------- App & Logging ----------------
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DATABASE_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "please_change_this")

# Logging to file + console
LOG_FILE = os.environ.get("LOG_FILE", "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(APP_NAME)

# ---------------- Extensions ----------------
db = SQLAlchemy(app)
CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ---------------- DB Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(320), unique=True, nullable=False)
    password_hash = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {"id": self.id, "email": self.email, "created_at": self.created_at.isoformat()}

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    headline = db.Column(db.Text)
    text = db.Column(db.Text, nullable=False)
    url = db.Column(db.Text)
    domain = db.Column(db.String(512))
    prediction = db.Column(db.String(32))
    confidence = db.Column(db.Float)
    class_probs = db.Column(db.Text)  # JSON string
    indicators = db.Column(db.Text)   # JSON string
    scores = db.Column(db.Text)       # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "headline": self.headline,
            "text": self.text,
            "url": self.url,
            "domain": self.domain,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "class_probs": json.loads(self.class_probs or "{}"),
            "indicators": json.loads(self.indicators or "{}"),
            "scores": json.loads(self.scores or "{}"),
            "timestamp": int(self.created_at.timestamp())
        }

# ---------------- Load Model & Vectorizer ----------------
model = None
vectorizer = None
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("Model and vectorizer loaded successfully.")
except Exception as e:
    logger.exception(f"Failed to load model/vectorizer: {e}")
    model = None
    vectorizer = None

# ---------------- Utilities ----------------
def create_db():
    """Create DB tables if not exist."""
    with app.app_context():
        db.create_all()
        logger.info("Database initialized.")

def generate_jwt(payload: dict, days=JWT_EXP_DAYS):
    exp = datetime.utcnow() + timedelta(days=days)
    payload_copy = payload.copy()
    payload_copy.update({"exp": exp})
    token = jwt.encode(payload_copy, JWT_SECRET, algorithm=JWT_ALGO)
    # pyjwt returns str in v2
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except Exception:
        return None

EMAIL_REGEX = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

# ---------------- Heuristics (kept from original) ----------------
def scan_for_advanced_indicators(text, domain=None):
    text_lower = text.lower()
    indicators = {"fake": [], "real": []}
    scores = {"fake": {}, "real": {}}

    headline_weight = 2
    body_weight = 1

    # Detect headlines from HTML or fallback to first 2 sentences
    headlines = re.findall(r"<h[1-2][^>]*>(.*?)</h[1-2]>", text, flags=re.IGNORECASE)
    body_text = re.sub(r"<.*?>", "", text)
    if not headlines:
        sentences = re.split(r'(?<=[.!?]) +', body_text)
        headlines = sentences[:2]

    emotional_words_list = [
        "shocking", "unbelievable", "secret", "miracle", "click here",
        "you won't believe", "must see", "breaking", "alert", "jaw-dropping",
        "amazing", "incredible", "sensational", "scandal", "reveal", "exposed"
    ]
    real_keywords = ["reported", "official", "study", "data", "research", "analysis", "according to"]

    def process_segment(segment, weight):
        seg_lower = segment.lower()
        for word in emotional_words_list:
            if word in seg_lower:
                indicators["fake"].append(word)
                scores["fake"][word] = scores["fake"].get(word, 0) + weight

        sentiment_score = abs(TextBlob(segment).sentiment.polarity)
        if sentiment_score > 0:
            indicators["fake"].append("emotional language")
            scores["fake"]["emotional language"] = scores["fake"].get("emotional language", 0) + sentiment_score * weight

        for word in real_keywords:
            if word in seg_lower:
                indicators["real"].append(word)
                scores["real"][word] = scores["real"].get(word, 0) + weight

    for h in headlines:
        process_segment(h, headline_weight)
    process_segment(body_text, body_weight)

    trustworthy_domains = ["bbc.com", "reuters.com", "apnews.com", "nytimes.com"]
    if domain and any(d in domain for d in trustworthy_domains):
        indicators["real"].append(f"trustworthy source: {domain}")
        scores["real"][f"trustworthy source: {domain}"] = 2

    if len(body_text.split()) < 50:
        indicators["fake"].append("very short content")
        scores["fake"]["very short content"] = 1.5
    elif len(body_text.split("\n")) > 2:
        indicators["real"].append("structured content")
        scores["real"]["structured content"] = 1.5

    return indicators, scores, headlines

def highlight_sentences_and_keywords(text, indicators, scores, headlines=None):
    text = re.sub(r'&', '&amp;', text)
    text = re.sub(r'<', '&lt;', text)
    text = re.sub(r'>', '&gt;', text)
    sentences = re.split(r'(?<=[.!?]) +', text)
    all_keywords = {kw: ("fake", scores["fake"].get(kw, 1)) for kw in indicators["fake"]}
    all_keywords.update({kw: ("real", scores["real"].get(kw, 1)) for kw in indicators["real"]})

    highlighted_sentences = []
    for sent in sentences:
        sentence_lower = sent.lower()
        sentence_bg = ""
        border_style = ""
        if headlines and any(h.strip().lower() in sentence_lower for h in headlines):
            border_style = 'border-left:4px solid #f39c12; padding-left:6px;'
        sentence_keywords = [kw for kw in all_keywords if kw.lower() in sentence_lower]
        if sentence_keywords:
            types = [all_keywords[kw][0] for kw in sentence_keywords]
            if types.count("fake") >= types.count("real"):
                sentence_bg = 'rgba(231, 76, 60, 0.1)'
            else:
                sentence_bg = 'rgba(52, 152, 219, 0.1)'
        for kw in sorted(sentence_keywords, key=len, reverse=True):
            kind, score = all_keywords[kw]
            intensity = min(score/5, 1)
            color = f'rgba(231, 76, 60, {0.3 + 0.7*intensity})' if kind=="fake" else f'rgba(52, 152, 219, {0.3 + 0.7*intensity})'
            tooltip = f'{kw} ({kind})'
            sent = re.sub(rf'\b({re.escape(kw)})\b',
                          f'<span style="background:{color}; font-weight:600;" title="{tooltip}">\\1</span>',
                          sent, flags=re.IGNORECASE)
        style_str = ""
        if sentence_bg: style_str += f'background:{sentence_bg}; padding:2px; border-radius:4px;'
        if border_style: style_str += border_style
        if style_str: sent = f'<span style="{style_str}">{sent}</span>'
        highlighted_sentences.append(sent)
    return " ".join(highlighted_sentences)

# ---------------- Routes ----------------
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "model not loaded"}), 500

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "no text provided"}), 400

    text = data["text"][:50000].strip()
    headline = data.get("headline", "").strip()
    url = data.get("url", "").strip()
    domain = urlparse(url).netloc if url else None

    if not text:
        return jsonify({"error": "empty text provided"}), 400

    try:
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        class_probs = {str(c): float(p) for c, p in zip(classes, proba)}
        label = "Real" if int(pred) == 1 else "Fake"
        confidence = float(proba.max())

        indicators, scores, headlines_detected = scan_for_advanced_indicators(text, domain=domain)
        headlines_list = [headline] if headline else headlines_detected

        result = {
            "prediction": label,
            "confidence": confidence,
            "class_probs": class_probs,
            "text": text,
            "headline": headline,
            "url": url,
            "domain": domain,
            "timestamp": int(time.time()),
            "indicators": indicators,
            "scores": scores,
            "headlines": headlines_list
        }

        # persist scan to DB
        scan = Scan(
            headline=headline,
            text=text,
            url=url,
            domain=domain,
            prediction=label,
            confidence=confidence,
            class_probs=json.dumps(class_probs),
            indicators=json.dumps(indicators),
            scores=json.dumps(scores)
        )
        db.session.add(scan)
        db.session.commit()

        logger.info(f"Prediction saved (scan_id={scan.id}) for domain={domain}")

        return jsonify(result)
    except Exception as e:
        logger.exception("Error during prediction")
        return jsonify({"error": f"model error: {str(e)}"}), 500

@app.route("/full-report")
def full_report():
    scan = Scan.query.order_by(Scan.created_at.desc()).first()
    if not scan:
        return "<h1>No analysis available yet.</h1><p>Please analyze an article first.</p>"

    pred = scan.to_dict()
    headline = pred.get("headline") or "Article"
    domain = pred.get("domain") or "Unknown"
    url = pred.get("url") or "Unknown"
    label = pred.get("prediction") or "Unknown"
    conf_pct = round(pred.get("confidence", 0) * 100, 1)
    fake_prob = round(pred.get("class_probs", {}).get("0", 0) * 100, 1)
    real_prob = round(pred.get("class_probs", {}).get("1", 0) * 100, 1)
    indicators = pred.get("indicators", {"fake": [], "real": []})
    scores = pred.get("scores", {"fake": {}, "real": {}})

    raw_text = pred.get("text", "")
    emotional_words = list(dict.fromkeys([w for w in indicators.get("fake", []) if w.lower() in raw_text.lower() and len(w) > 2]))
    domain_trusted = any(d in (domain or "").lower() for d in ["bbc.com","reuters.com","apnews.com","nytimes.com"])

    combined = {**{k: float(v) for k, v in scores.get("fake", {}).items()}, **{k: float(v) for k, v in scores.get("real", {}).items()}}
    sorted_combined = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    top_keywords = [k for k, v in sorted_combined[:8]]

    highlighted_html = highlight_sentences_and_keywords(raw_text, indicators, scores, pred.get("headlines", []))

    fake_keywords = list(scores.get("fake", {}).keys())
    fake_scores = [float(scores["fake"].get(k, 0)) for k in fake_keywords]
    real_keywords = list(scores.get("real", {}).keys())
    real_scores = [float(scores["real"].get(k, 0)) for k in real_keywords]

    # render same template as before (kept for compatibility)
    return render_template_string("""<!doctype html>... (omitted for brevity) ...</html>""",
        # note: you should paste your full HTML template here or move it to templates/
        headline=headline, domain=domain, url=url, label=label, conf_pct=conf_pct,
        fake_prob=fake_prob, real_prob=real_prob, emotional_words=emotional_words,
        domain_trusted=domain_trusted, top_keywords=top_keywords, fake_keywords=fake_keywords,
        real_keywords=real_keywords, fake_scores=fake_scores, real_scores=real_scores,
        highlighted_html=highlighted_html)

@app.route("/history")
def history():
    # simple HTML list of last 20 scans
    scans = Scan.query.order_by(Scan.created_at.desc()).limit(20).all()
    if not scans:
        return "<h1>No scans yet</h1><p>Analyze some articles first.</p>"
    html_items = ""
    for scan in scans:
        pred = scan.to_dict()
        headline = pred.get("headline") or "No headline"
        prediction = pred.get("prediction", "Unknown")
        confidence = round(pred.get("confidence", 0) * 100, 1)
        color = "#e74c3c" if prediction.lower() == "fake" else "#3498db"
        html_items += f"""
        <div style="background:#fff; border-radius:10px; padding:15px; margin-bottom:12px; box-shadow:0 6px 18px rgba(0,0,0,0.1);">
            <h2 style="margin:0; color:{color};">{headline}</h2>
            <p><strong>Prediction:</strong> {prediction} | <strong>Confidence:</strong> {confidence}%</p>
        </div>"""
    return f"<html><body><h1>Last 20 Scanned Headlines</h1>{html_items}</body></html>"

@app.route("/history-json")
def history_json():
    scans = Scan.query.order_by(Scan.created_at.desc()).limit(50).all()
    return jsonify([s.to_dict() for s in scans])

# ---------------- Auth ----------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required."}), 400
    if not EMAIL_REGEX.match(email):
        return jsonify({"success": False, "message": "Invalid email address."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({"success": False, "message": "Email already registered."}), 400

    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    user = User(email=email, password_hash=hashed_pw)
    db.session.add(user)
    db.session.commit()

    token = generate_jwt({"sub": user.email, "id": user.id})
    logger.info(f"New user registered: {email}")
    return jsonify({"success": True, "message": "Registration successful.", "token": token, "email": user.email})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not email or not password:
        return jsonify({"success": False, "message": "Email and password required."}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "User not found."}), 404

    if not bcrypt.checkpw(password.encode("utf-8"), user.password_hash):
        return jsonify({"success": False, "message": "Incorrect password."}), 401

    token = generate_jwt({"sub": user.email, "id": user.id})
    logger.info(f"User logged in: {email}")
    return jsonify({"success": True, "message": "Login successful.", "token": token, "email": user.email})

# ---------------- Health & Info ----------------
@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None, "vectorizer_loaded": vectorizer is not None})

# ---------------- Start ----------------
if __name__ == "__main__":
    create_db()
    logger.info(f"Starting {APP_NAME} on port {PORT}")
    # Use the assigned port (for Render it will set PORT env var)
    app.run(host="0.0.0.0", port=PORT)
