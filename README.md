# Fake News Detector 101

**Fake News Detector 101** is an AI-powered browser extension that analyzes any webpage in real time to detect misleading, biased, or false news. It provides clear explanations, confidence scores, and trust indicators — making fact-checking effortless for everyone.

---

## 🔵 Features

* ✅ **Scan any website** for fake or real news
* ✅ **Instant trust score** (tone, sentiment, domain credibility)
* ✅ **Explainable AI insights** — highlighted keywords, tone, and evidence
* ✅ **Privacy-first** — no tracking, no personal data collection
* ✅ **Compatible with major news sources** (CNN, BBC, Reuters, The Guardian, etc.)

---

## 🔵 Official Links

* **Website:** [https://www.fakenewsdetector101.com](https://www.fakenewsdetector101.com)
* **Backend API:** [https://fakenewsdetector-zjzs.onrender.com](https://fakenewsdetector-zjzs.onrender.com)
* **Source Code:** [https://github.com/JazL0T/FakeNewsDetector](https://github.com/JazL0T/FakeNewsDetector)
* **Privacy Policy**
* **Terms of Service**
* **Contact Page**

---

## 🔵 How It Works

1. Install the **Fake News Detector 101** extension.
2. Open any webpage or online article.
3. Click the extension icon to **Scan**.
4. The AI analyzes the text (English + Malay) and returns:

   * 🟢 **Real** — reliable
   * 🔴 **Fake** — misleading
   * 🟡 **Uncertain** — mixed indicators
5. Review the detailed output: **confidence score**, **tone**, **keywords**, **evidence**, and **source trust rating**.

---

## 🔵 Technical Overview

| Component      | Description                                     |
| -------------- | ----------------------------------------------- |
| **Backend**    | Python (Flask), SQLite, TextBlob, Scikit-learn  |
| **Model**      | Logistic Regression + TF-IDF heuristic analysis |
| **Frontend**   | Chrome Extension (Manifest V3)                  |
| **Deployment** | Render Cloud                                    |
| **Security**   | HTTPS, JWT authentication, `.env` protection    |

---

## 🔵 Security & Privacy

* No personal data stored or shared
* No tracking or analytics
* No browsing history collected
* All communication secured via **HTTPS**
* Backend is open-source for transparency

---

## 🔵 Dataset Usage & Credits

This project uses publicly available datasets for academic research.
Please credit the original curators:

1. **Zolkepli, Husein — “Malay-Dataset”**
   Text corpus for Bahasa Malaysia
   [https://github.com/huseinzol05/Malay-Dataset](https://github.com/huseinzol05/Malay-Dataset)

2. **Zolkepli, Husein — “Malaya”**
   Natural Language Toolkit for Bahasa Malaysia
   [https://github.com/huseinzol05/Malaya](https://github.com/huseinzol05/Malaya)

3. **Abaghyangor — “Fake News Dataset” (Kaggle)**
   [https://www.kaggle.com/datasets/abaghyangor/fake-news-dataset](https://www.kaggle.com/datasets/abaghyangor/fake-news-dataset)

---

## 🔵 Publication & Academic Reference

The work done in this project is part of the following publication:

**"A Benchmark Evaluation Study for Malay Fake News Classification Using Neural Network Architectures"**
Published in **Kazan Digital Week 2020**, *Methodical and Informational Science Journal*,
Vestnik NTsBZhD (4), pp. 5–13, 2020.

**Links:**

* PDF: [https://ncbgd.tatarstan.ru/rus/file/pub/pub_2610566.pdf](https://ncbgd.tatarstan.ru/rus/file/pub/pub_2610566.pdf)
* Journal Site: [http://www.vestnikncbgd.ru/index.php?id=1&lang=en](http://www.vestnikncbgd.ru/index.php?id=1&lang=en)
* Event: [https://kazandigitalweek.com/](https://kazandigitalweek.com/)
* Related Work (GitHub): [https://github.com/AsyrafAzlan/malay-fake-news-classification.git](https://github.com/AsyrafAzlan/malay-fake-news-classification.git)

---

## 🔵 Developer Setup

```bash
# 1. Clone the repository
git clone https://github.com/JazL0T/FakeNewsDetector.git

# 2. Enter backend folder
cd FakeNewsDetector/backend

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the backend
python app.py
```

---

If you'd like, I can help you add badges, screenshots, or a roadmap section.
