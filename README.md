# 🧠 Fake News Detector 101

**Fake News Detector 101** is a smart browser extension powered by AI and linguistic analysis that helps you instantly detect misleading, biased, or false news articles online — making fact-checking effortless for everyone.

---

## 🔵 Features

✅ **Scan any website** to detect fake or real news  
✅ **Instant trust score** — based on tone, sentiment, and domain credibility  
✅ **Explainable AI insights** — highlights keywords, tone, and evidence from the text  
✅ **Privacy-first design** — no tracking, no personal data collection  
✅ **Works with major news outlets** — CNN, BBC, Reuters, The Guardian, and more  

---

## 🔵 Official Links

- **Website:** [https://www.fakenewsdetector101.com](https://www.fakenewsdetector101.com/)
- **Privacy Policy:** [https://www.fakenewsdetector101.com/privacy.php](https://www.fakenewsdetector101.com/privacy.php)
- **Terms of Service:** [https://www.fakenewsdetector101.com/terms.php](https://www.fakenewsdetector101.com/terms.php)
- **Contact:** [https://www.fakenewsdetector101.com/contact.php](https://www.fakenewsdetector101.com/contact.php)
- **Backend API:** [https://fakenewsdetector-zjzs.onrender.com](https://fakenewsdetector-zjzs.onrender.com)
- **Source Code (GitHub):** [https://github.com/JazL0T/FakeNewsDetector](https://github.com/JazL0T/FakeNewsDetector)

---

## 🔵 How It Works

1. Install the **Fake News Detector 101** Chrome extension.  
2. Visit any online article or webpage.  
3. Click the extension icon to **scan** the content.  
4. The AI analyzes the text and returns:
   - 🟢 **Real** — reliable and factual  
   - 🔴 **Fake** — misleading or suspicious  
   - 🟡 **Uncertain** — mixed indicators (verify further)  
5. View a detailed **confidence score**, **tone**, and **trustability analysis**.

---

## 🔵 Technical Overview

| Component | Description |
|------------|-------------|
| **Backend** | Flask (Python), SQLite, TextBlob, Scikit-learn |
| **Frontend** | Chrome Extension (Manifest V3, HTML, CSS, JS) |
| **Deployment** | Render Cloud (Python API) |
| **Model** | Logistic Regression + TF-IDF Heuristic Analysis |
| **Security** | JWT-based auth, HTTPS enforced, .env protected |

---

## 🔵 Security & Privacy

- No personal data is stored, shared, or sold.  
- All communication uses secure **HTTPS**.  
- Authentication uses **JWT** (JSON Web Tokens).  
- Sensitive credentials are kept in `.env` (not public in GitHub).  
- Backend code is open for transparency.

---

## 🔵 Developer Setup 

If you want to run your own version locally:

```bash
# 1. Clone the repository
git clone https://github.com/JazL0T/FakeNewsDetector.git

# 2. Enter backend folder
cd FakeNewsDetector/backend

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask backend
python app.py
