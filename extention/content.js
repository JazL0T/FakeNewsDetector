// content.js
console.log("Fake News Detector content.js loaded ✅");

// ---------------- Helpers ----------------
function extractHeadline() {
  const headlineSelectors = ["h1", ".headline", "article h1", "main h1"];
  for (const sel of headlineSelectors) {
    const h = document.querySelector(sel);
    if (h?.innerText?.trim()) return h.innerText.trim();
  }
  return "";
}

function extractArticleText() {
  const url = window.location.hostname;
  let selectors = [];

  // --- Site-specific rules ---
  if (url.includes("bbc.com")) {
    selectors = ["article p", ".ssrcss-uf6wea-RichTextComponentWrapper p"];
  } else if (url.includes("reuters.com")) {
    selectors = ["article p", 'div[data-testid="paragraph"] p', ".article-content p"];
  } else if (url.includes("cnn.com")) {
    selectors = [".article__content p", ".zn-body__paragraph"];
  } else if (url.includes("nytimes.com")) {
    selectors = ["section[name='articleBody'] p"];
  } else if (url.includes("theguardian.com")) {
    selectors = [".article-body-commercial-selector p", ".content__article-body p"];
  } else if (url.includes("aljazeera.com")) {
    selectors = [".wysiwyg p", "article p"];
  } else {
    // Generic fallback
    selectors = [
      "article p",
      "main p",
      'div[class*="article"] p',
      'div[class*="content"] p'
    ];
  }

  let collected = [];

  // Loop all selectors and gather text
  for (const sel of selectors) {
    const nodes = document.querySelectorAll(sel);
    nodes.forEach(p => {
      const text = p.innerText.trim();
      if (text && text.length > 20 && !/advertisement|subscribe|sign up/i.test(text)) {
        collected.push(text);
      }
    });
    if (collected.length > 5) break;
  }

  // Fallback: grab all <p> tags if too little was collected
  if (collected.length < 5) {
    document.querySelectorAll("p").forEach(p => {
      const text = p.innerText.trim();
      if (text && text.length > 20 && !/advertisement|subscribe|sign up/i.test(text)) {
        collected.push(text);
      }
    });
  }

  return collected.join("\n\n");
}

// ---------------- Banner ----------------
function createBanner(contentHtml) {
  if (document.querySelector("#fnd-banner")) return;

  const banner = document.createElement("div");
  banner.id = "fnd-banner";
  Object.assign(banner.style, {
    position: "fixed",
    top: "20px",
    left: "50%",
    transform: "translateX(-50%) translateY(-120%)",
    width: "90%",
    maxWidth: "450px",
    zIndex: 2147483647,
    padding: "18px 22px",
    fontSize: "16px",
    fontWeight: "500",
    textAlign: "center",
    boxShadow: "0 8px 25px rgba(0,0,0,0.35)",
    borderRadius: "14px",
    background: "#2c3e50",
    color: "#fff",
    fontFamily: "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
    transition: "transform 0.5s ease, opacity 0.5s ease",
    opacity: 0,
    pointerEvents: "auto"
  });
  banner.innerHTML = contentHtml;

  const closeBtn = document.createElement("button");
  closeBtn.innerText = "×";
  Object.assign(closeBtn.style, {
    position: "absolute",
    right: "14px",
    top: "10px",
    background: "transparent",
    color: "#fff",
    border: "none",
    fontSize: "22px",
    cursor: "pointer"
  });
  closeBtn.addEventListener("click", () => banner.remove());
  banner.appendChild(closeBtn);

  document.body.appendChild(banner);
  setTimeout(() => {
    banner.style.transform = "translateX(-50%) translateY(0)";
    banner.style.opacity = "1";
  }, 50);
  return banner;
}

// ---------------- Message Listener ----------------
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "scanPage") {
    (async () => {
      try {
        const headline = extractHeadline();
        let articleText = extractArticleText();

        if (!articleText || articleText.split(/\s+/).length < 20) {
          sendResponse({ error: "Not enough article text found" });
          return;
        }

        // ✅ Use hosted API instead of local
        const apiUrl = "https://fakenewsdetector101.onrender.com/predict";

        const resp = await fetch(apiUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            text: articleText,
            headline,
            url: window.location.href
          })
        }).catch(err => {
          console.error("API fetch failed:", err);
          sendResponse({ error: "Could not reach prediction server." });
        });

        if (!resp || !resp.ok) throw new Error(`status ${resp?.status || "no response"}`);
        const result = await resp.json();

        const enrichedResult = {
          ...result,
          headline,
          text: articleText,
          url: window.location.href
        };

        chrome.storage.local.set({ latestPrediction: enrichedResult });

        // --- Banner ---
        const label = (result.prediction || "").toLowerCase();
        const conf = (result.confidence * 100).toFixed(1);
        const fakeProb = ((result.class_probs?.["0"] ?? 0) * 100).toFixed(1);
        const realProb = ((result.class_probs?.["1"] ?? 0) * 100).toFixed(1);

        const bannerHtml = `
          <div style="padding:16px; border-radius:12px; background:#2c3e50; color:#fff; font-weight:600;">
            <div style="font-size:18px; margin-bottom:6px;">
              ${headline || "Article"}: ${label.includes("fake") ? "⚠️ FAKE NEWS" : "✅ REAL NEWS"}
            </div>
            <div>Overall Confidence: <b>${conf}%</b></div>
            <div style="margin-top:10px; height:18px; background:#34495e; border-radius:10px; display:flex; overflow:hidden;">
              <div style="flex:0 0 ${Math.max(fakeProb,10)}%; background:#e74c3c; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; border-radius:10px 0 0 10px;">
                Fake ${fakeProb}%
              </div>
              <div style="flex:0 0 ${Math.max(realProb,10)}%; background:#3498db; display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; border-radius:0 10px 10px 0;">
                Real ${realProb}%
              </div>
            </div>
          </div>
        `;
        createBanner(bannerHtml);

        sendResponse({ success: true, result: enrichedResult });
      } catch (e) {
        console.error("Scan error:", e);
        sendResponse({ error: e.message || "Unknown error" });
      }
    })();

    return true;
  }
});
