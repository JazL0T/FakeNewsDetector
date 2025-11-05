// ===========================================================
// Fake News Detector — Popup (Polished Labels + Clear Trust Context + Smart Summary)
// ===========================================================

const API_BASE = "https://fakenewsdetector-zjzs.onrender.com";
const KEY_JWT = "fnd_jwt";
const KEY_USER = "fnd_username";
const KEY_LASTSCAN = "fnd_lastScan";

// --- DOM Elements ---
const tabScan = document.getElementById("tab-scan");
const tabHistory = document.getElementById("tab-history");
const viewScan = document.getElementById("view-scan");
const viewHistory = document.getElementById("view-history");

const scanBtn = document.getElementById("scan-btn");
const fullReportBtn = document.getElementById("full-report-btn");
const scanStatus = document.getElementById("scan-status");
const spinner = document.getElementById("loading-spinner");
const summary = document.getElementById("result-summary");
const summaryText = document.getElementById("summary-text");
const resCard = document.getElementById("results");

const resHeadline = document.getElementById("res-headline");
const resUrl = document.getElementById("res-url");
const resPrediction = document.getElementById("res-prediction");
const resConfidence = document.getElementById("res-confidence");
const resDomain = document.getElementById("res-domain");
const resTrustScore = document.getElementById("res-trust-score");
const resTrustCat = document.getElementById("res-trust-cat");

const historyList = document.getElementById("history-list");
const historyHint = document.getElementById("history-hint");
const userStatus = document.getElementById("user-status");
const loginTip = document.getElementById("login-tip");
const loginLink = document.getElementById("login-link");
const logoutBtn = document.getElementById("logout-btn");
const websiteBtn = document.getElementById("visit-website");

// --- Helpers ---
function showStatus(msg, isError = false) {
  scanStatus.textContent = msg;
  scanStatus.style.color = isError ? "#e53935" : "#1565c0";
}

function toggleLoading(isLoading) {
  spinner.classList.toggle("hidden", !isLoading);
  scanBtn.disabled = isLoading;
  scanBtn.textContent = isLoading ? "⏳ Scanning..." : "Scan this page";
}

function fmtPct(n) {
  const num = Number(n);
  return Number.isNaN(num) ? "0%" : (num * 100).toFixed(1) + "%";
}

async function getAuth() {
  const v = await chrome.storage.local.get([KEY_JWT, KEY_USER]);
  return { jwt: v[KEY_JWT], username: v[KEY_USER] };
}

function fadeIn(el) {
  el.classList.remove("hidden");
  el.style.opacity = 0;
  el.style.transition = "opacity 0.3s ease";
  requestAnimationFrame(() => (el.style.opacity = 1));
}

function show(view) {
  [viewScan, viewHistory].forEach(v => v && v.classList.add("hidden"));
  if (view) fadeIn(view);
}

function activate(tab) {
  [tabScan, tabHistory].forEach(t => t?.classList.remove("active"));
  tab?.classList.add("active");
}

// --- TAB NAVIGATION ---
tabScan?.addEventListener("click", () => {
  activate(tabScan);
  show(viewScan);
});

tabHistory?.addEventListener("click", async () => {
  activate(tabHistory);
  show(viewHistory);
  await renderHistory();
});

// --- SCAN ---
scanBtn?.addEventListener("click", () => performScan());

async function performScan() {
  const { jwt, username } = await getAuth();
  toggleLoading(true);
  summary.classList.add("hidden");
  resCard.classList.add("hidden");
  fullReportBtn.classList.add("hidden");
  showStatus("Extracting article...");

  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) {
    toggleLoading(false);
    return showStatus("No active tab found.", true);
  }

  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["content.js"] });
  } catch (err) {
    console.warn("content.js injection skipped", err);
  }

  try {
    const result = await chrome.tabs.sendMessage(tab.id, { action: "scanPage" }).catch(() => null);
    if (!result) throw new Error("Could not connect to content script.");
    if (result.error) throw new Error(result.error);

    const { headline, text, url } = result.result;
    const headers = { "Content-Type": "application/json" };
    if (jwt) headers["Authorization"] = `Bearer ${jwt}`;

    showStatus("Analyzing content...");
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers,
      body: JSON.stringify({ headline, text, url })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");

    renderResults(url, headline, data);
    chrome.storage.local.set({ [KEY_LASTSCAN]: data });

    if (username && jwt) fullReportBtn.classList.remove("hidden");
    showStatus("");
  } catch (err) {
    showStatus(`⚠️ ${err.message}`, true);
  } finally {
    toggleLoading(false);
  }
}

// --- RESULTS ---
function renderResults(url, headline, data) {
  resUrl.innerHTML = url ? `<a href="${url}" target="_blank" class="result-link">${url}</a>` : "—";
  resHeadline.textContent = headline || "—";

  const pred = data.prediction;
  resPrediction.textContent = pred;
  resPrediction.className = "result-badge " + (pred === "Fake" ? "fake" : "real");

  const conf = fmtPct(data.confidence);
resConfidence.textContent = conf;

// Get the confidence bar element
const bar = document.getElementById("confidence-fill");

if (bar) {
  // Set width
  bar.style.width = conf;

  // Reset old colors
  bar.style.backgroundColor = "";

  // Color based on prediction result
  if (data.prediction === "Fake") {
    bar.style.backgroundColor = "#e53935"; // Red
  } else if (data.prediction === "Real") {
    bar.style.backgroundColor = "#2e7d32"; // Green
  } else {
    bar.style.backgroundColor = "#f9a825"; // Yellow (Uncertain)
  }
}

  const t = data.trustability || {};
  resDomain.textContent = t.domain || "—";
  resTrustScore.textContent = t.trust_score ?? "—";
  resTrustCat.textContent = t.category || "—";
  resTrustCat.className =
    "result-badge " +
    (t.category === "Trusted" ? "real" : t.category === "Suspicious" ? "fake" : "uncertain");

  // Smart summary messages
  let summaryMsg = "";
  let summaryColor = "#e3f2fd";

  if (pred === "Fake" && t.category === "Trusted") {
    summaryMsg = `🟠 The content looks suspicious, but the source (${t.domain}) is reputable. Possibly satire or opinion-based.`;
    summaryColor = "#fff3cd";
  } else if (pred === "Fake" && t.category === "Suspicious") {
    summaryMsg = `🔴 Both the article and its source (${t.domain || "unknown"}) seem unreliable. Proceed with caution.`;
    summaryColor = "#ffebee";
  } else if (pred === "Real" && t.category === "Trusted") {
    summaryMsg = `🟢 Legitimate article from a verified, trusted outlet (${t.domain}).`;
    summaryColor = "#e8f5e9";
  } else if (pred === "Real" && t.category === "Uncertain") {
    summaryMsg = `🟡 The writing seems factual, but the source (${t.domain}) is unverified. Check other outlets.`;
    summaryColor = "#fffde7";
  } else {
    summaryMsg = `⚪ Mixed indicators — verify before trusting completely.`;
    summaryColor = "#f5f5f5";
  }

  summary.style.backgroundColor = summaryColor;
  summaryText.textContent = summaryMsg;

  fadeIn(summary);
  fadeIn(resCard);
}

// --- FULL REPORT ---
fullReportBtn?.addEventListener("click", async () => {
  const { jwt } = await getAuth();
  if (!jwt) return showStatus("🔒 Please log in to view reports.", true);

  try {
    const res = await fetch(`${API_BASE}/get-history`, { headers: { Authorization: `Bearer ${jwt}` } });
    const data = await res.json();
    if (!res.ok || !data.history?.length) return showStatus("⚠️ No reports found.", true);

    const latest = data.history[0];
    chrome.tabs.create({ url: `${API_BASE}/get-report/${latest.id}?token=${encodeURIComponent(jwt)}` });
  } catch {
    showStatus("⚠️ Failed to open full report.", true);
  }
});

// --- HISTORY ---
async function renderHistory() {
  const { jwt, username } = await getAuth();

  if (!jwt || !username) {
    historyList.innerHTML = "";
    historyHint.style.display = "block";
    historyHint.textContent = "🔒 Log in to view your past scans.";
    return;
  }

  historyHint.style.display = "none";
  historyList.innerHTML = "<div class='muted'>Loading your history...</div>";

  try {
    const res = await fetch(`${API_BASE}/get-history`, {
      headers: { Authorization: `Bearer ${jwt}` }
    });
    const data = await res.json();

    if (!res.ok || !data.history?.length) {
      historyList.innerHTML = "<div class='muted'>No scans yet. Start scanning to build your history!</div>";
      return;
    }

    historyList.innerHTML = "";
    data.history.slice(0, 10).forEach(item => {
      const trust = item.trustability || {};
      const cat = (trust.category || "").toLowerCase();
      const catColor = cat === "trusted" ? "#2e7d32" : cat === "suspicious" ? "#e53935" : "#f9a825";

      const predColor = item.prediction === "Fake" ? "#e53935" : "#2e7d32";
      const predEmoji = item.prediction === "Fake" ? "🔴" : "🟢";
      const predBadge = `
        <span style="
          background:${predColor}15;
          color:${predColor};
          font-weight:600;
          padding:3px 8px;
          border-radius:6px;
          font-size:13px;
        ">${predEmoji} ${item.prediction}</span>`;

      const sourceNote =
        cat === "trusted"
          ? " — trusted news outlet"
          : cat === "uncertain"
          ? " — source credibility unclear"
          : cat === "suspicious"
          ? " — low-credibility or clickbait site"
          : "";

      const node = document.createElement("div");
      node.className = "hist-card";
      node.innerHTML = `
        <div class="hist-header">${item.headline || "(No headline)"}</div>
        <div class="hist-body">
          <div class="hist-info">
            ${predBadge} <span style="color:#555;">(${fmtPct(item.confidence)})</span> -
            <span style="color:${catColor};font-weight:600;">${trust.category}${sourceNote}</span>
          </div>
          <div class="hist-time">🕒 ${new Date(item.timestamp).toLocaleString()}</div>
        </div>
      `;
      node.addEventListener("click", () =>
        chrome.tabs.create({
          url: `${API_BASE}/get-report/${item.id}?token=${encodeURIComponent(jwt)}`
        })
      );
      historyList.appendChild(node);
    });
  } catch (err) {
    console.error(err);
    historyList.innerHTML = "<div class='muted'>⚠️ Failed to load history.</div>";
  }
}

// --- LOGIN / LOGOUT ---
loginLink?.addEventListener("click", () =>
  chrome.tabs.create({ url: chrome.runtime.getURL("login.html") })
);

logoutBtn?.addEventListener("click", async () => {
  await chrome.storage.local.remove([KEY_JWT, KEY_USER]);
  userStatus.textContent = "🔒 Guest mode";
  loginTip.style.display = "block";
  logoutBtn.classList.add("hidden");
  loginLink.classList.remove("hidden");
  fullReportBtn.classList.add("hidden");
  historyList.innerHTML = "";
  historyHint.style.display = "block";
  historyHint.textContent = "🔒 Log in to view your past scans.";
});

// --- WEBSITE BUTTON ---
websiteBtn?.addEventListener("click", () => {
  chrome.tabs.create({ url: "https://www.fakenewsdetector101.com/" });
});

// --- INIT ---
document.addEventListener("DOMContentLoaded", async () => {
  // ✅ Privacy Policy Check (only open once)
  const { fnd_privacyAccepted, fnd_privacyShown } = await chrome.storage.local.get([
    "fnd_privacyAccepted",
    "fnd_privacyShown"
  ]);

  // If user hasn’t accepted but policy not shown yet
  if (!fnd_privacyAccepted) {
    if (!fnd_privacyShown) {
      await chrome.storage.local.set({ fnd_privacyShown: true });
      chrome.tabs.create({ url: chrome.runtime.getURL("privacy_policy.html") });
    }
    window.close();
    return;
  }

  // Normal Extension Startup
  const { jwt, username } = await getAuth();
  const loggedIn = jwt && username;

  userStatus.textContent = loggedIn ? `👤 Logged in as ${username}` : "🔓 Guest mode";
  loginTip.style.display = loggedIn ? "none" : "block";

  if (!loggedIn) {
    loginLink.classList.remove("hidden");
    logoutBtn.classList.add("hidden");
  } else {
    loginLink.classList.add("hidden");
    logoutBtn.classList.remove("hidden");
  }

  const { [KEY_LASTSCAN]: last } = await chrome.storage.local.get(KEY_LASTSCAN);
  if (last) renderResults(last.url, last.headline, last);

  activate(tabScan);
  show(viewScan);
});