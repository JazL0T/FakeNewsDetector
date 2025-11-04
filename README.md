# Fake News Detector 101

**Fake News Detector 101** is a smart browser extension that uses AI and linguistic analysis to detect misleading or false news online — helping you verify information instantly.

---

## Features

✅ **Scan any website** to detect fake or real news  
✅ **Instant trust score** — based on tone, sentiment, and domain reputation  
✅ **Explainable AI insights** — highlights key evidence in the article  
✅ **100% privacy-first** — no tracking, no personal data collection  

---

## Website

**Official Website:** [https://www.fakenewsdetector101.com](https://www.fakenewsdetector101.com/)  
**Privacy Policy:** [https://www.fakenewsdetector101.com/privacy.php](https://www.fakenewsdetector101.com/privacy.php)  
**Terms of Service:** [https://www.fakenewsdetector101.com/terms.php](https://www.fakenewsdetector101.com/terms.php)

**Contact:** [https://www.fakenewsdetector101.com/contact.php](https://www.fakenewsdetector101.com/contact.php)

---

## How It Works

1. Install the **Fake News Detector 101** Chrome extension  
2. Click the extension icon while reading an article  
3. The AI scans the content and returns:
   - 🟢 **Real** — reliable and factual  
   - 🔴 **Fake** — misleading or suspicious  
   - 🟡 **Uncertain** — mixed indicators; verify further  
4. Get a **confidence score** and **trustability breakdown**

---

## Technical Overview

- **Backend:** Flask (Python) + SQLite + TextBlob + Scikit-learn  
- **Frontend:** Chrome Extension (Manifest V3, HTML, CSS, JS)  
- **Deployment:** Render Cloud (Python backend API)  
- **AI Model:** Logistic Regression + Heuristic Analysis (TF-IDF)

---

## Security

- No user data is stored or tracked.  
- JWT-based authentication is used for secure logins.  
- Environment variables (`.env`) protect sensitive keys and API secrets.  
- HTTPS enforced for API endpoints.

---

## Developer Setup (Optional)

If you want to run your own version:

```bash
git clone https://github.com/JazL0T/FakeNewsDetector.git
cd FakeNewsDetector/backend
pip install -r requirements.txt
python app.py

