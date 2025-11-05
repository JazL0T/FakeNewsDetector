// filename: popup.js
// ===========================================================
// Fake News Detector — Popup (Polished Labels + Clear Trust Context + Smart Summary)
// Firefox-compatible (Promise wrappers + scripting shim fallback)
// ===========================================================

const API_BASE = "https://fakenewsdetector-zjzs.onrender.com";
const KEY_JWT = "fnd_jwt";
const KEY_USER = "fnd_username";
const KEY_LASTSCAN = "fnd_lastScan";

// ---- Promise helpers (FF MV2 uses callbacks) ----
const pTabsQuery     = (q)    => new Promise((res) => chrome.tabs.query(q, res));
const pTabsCreate    = (opt)  => new Promise((res) => chrome.tabs.create(opt, res));
const pStorageGet    = (k)    => new Promise((res) => chrome.storage.local.get(k, res));
const pStorageSet    = (obj)  => new Promise((res) => chrome.storage.local.set(obj, res));
const pSendMessage   = (id,m) => new Promise((res) => chrome.tabs.sendMessage(id, m, res));

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
  const v = await pStorageGet([KEY_JWT, KEY_USER]);
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

async function injectContentScript(tabId) {
  // Prefer MV3 API if available (Chrome); compat.js shims this for FF
  if (chrome.scripting && chrome.scripting.executeScript) {
    try {
      await chrome.scripting.executeScript({ target: { tabId }, files: ["content.js"] });
      return true;
    } catch (e) {
      console.warn("MV3 inject failed, trying MV2:", e);
    }
  }
  // Fallback for Firefox MV2
  return new Promise((resolve) => {
    chrome.tabs.executeScript(tabId, { file: "content.js" }, () => {
      if (chrome.runtime.lastError) {
        console.warn("MV2 inject error:", chrome.runtime.lastError);
        resolve(false);
      } else {
        resolve(true);
      }
    });
  });
}

async function performScan() {
  const { jwt, username } = await getAuth();
  toggleLoading(true);
  summary.classList.add("hidden");
  resCard.classList.add("hidden");
  fullReportBtn.classList.add("hidden");
  showStatus("Extracting article...");

  const [tab] = await pTabsQuery({ active: true, currentWindow: true });
  if (!tab?.id) {
    toggleLoading(false);
    return showStatus("No active tab found.", true);
  }

  // Try injecting content script (safe if already injected)
  await injectContentScript(tab.id);

  try {
    const result = await pSendMessage(tab.id, { action: "scanPage" }) || null;
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
    pStorageSet({ [KEY_LASTSCAN]: data });

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
  if (url) {
  const a = document.createElement("a");
  a.href = url;
  a.target = "_blank";
  a.className = "result-link";
  a.textContent = url;
  resUrl.innerHTML = ""; // clear previous
  resUrl.appendChild(a);
} else {
  resUrl.textContent = "—";
}

  resHeadline.textContent = headline || "—";

  const pred = data.prediction;
  resPrediction.textContent = pred;
  resPrediction.className = "result-badge " + (pred === "Fake" ? "fake" : "real");

  const conf = fmtPct(data.confidence);
  resConfidence.textContent = conf;

  const bar = document.getElementById("confidence-fill");
  if (bar) {
    bar.style.width = conf;
    bar.style.backgroundColor = "";
    if (data.prediction === "Fake")      bar.style.backgroundColor = "#e53935";
    else if (data.prediction === "Real") bar.style.backgroundColor = "#2e7d32";
    else                                  bar.style.backgroundColor = "#f9a825";
  }

  const t = data.trustability || {};
  resDomain.textContent = t.domain || "—";
  resTrustScore.textContent = t.trust_score ?? "—";
  resTrustCat.textContent = t.category || "—";
  resTrustCat.className =
    "result-badge " +
    (t.category === "Trusted" ? "real" : t.category === "Suspicious" ? "fake" : "uncertain");

  // Smart summary
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
    await pTabsCreate({ url: `${API_BASE}/get-report/${latest.id}?token=${encodeURIComponent(jwt)}` });
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
      historyList.innerHTML =
        "<div class='muted'>No scans yet. Start scanning to build your history!</div>";
      return;
    }

    historyList.innerHTML = "";

    data.history.slice(0, 10).forEach(item => {
      const trust = item.trustability || {};
      const cat = (trust.category || "").toLowerCase();
      const catColor =
        cat === "trusted" ? "#2e7d32" :
        cat === "suspicious" ? "#e53935" :
        "#f9a825";

      const predColor = item.prediction === "Fake" ? "#e53935" : "#2e7d32";
      const predEmoji = item.prediction === "Fake" ? "🔴" : "🟢";

      // Card wrapper
      const node = document.createElement("div");
      node.className = "hist-card";

      // Headline
      const header = document.createElement("div");
      header.className = "hist-header";
      header.textContent = item.headline || "(No headline)";
      node.appendChild(header);

      // Body container
      const body = document.createElement("div");
      body.className = "hist-body";

      // Info row
      const info = document.createElement("div");
      info.className = "hist-info";

      // Prediction badge
      const predSpan = document.createElement("span");
      predSpan.style.background = `${predColor}15`;
      predSpan.style.color = predColor;
      predSpan.style.fontWeight = "600";
      predSpan.style.padding = "3px 8px";
      predSpan.style.borderRadius = "6px";
      predSpan.style.fontSize = "13px";
      predSpan.textContent = `${predEmoji} ${item.prediction}`;
      info.appendChild(predSpan);

      // Confidence
      const confSpan = document.createElement("span");
      confSpan.style.color = "#555";
      confSpan.textContent = ` (${fmtPct(item.confidence)}) - `;
      info.appendChild(confSpan);

      // Trust category + note
      const catSpan = document.createElement("span");
      catSpan.style.color = catColor;
      catSpan.style.fontWeight = "600";

      let sourceNote = "";
      if (cat === "trusted") sourceNote = " — trusted news outlet";
      else if (cat === "uncertain") sourceNote = " — source credibility unclear";
      else if (cat === "suspicious") sourceNote = " — low-credibility or clickbait site";

      catSpan.textContent = `${trust.category || "Unknown"}${sourceNote}`;
      info.appendChild(catSpan);

      body.appendChild(info);

      // Timestamp
      const time = document.createElement("div");
      time.className = "hist-time";
      time.textContent = `🕒 ${new Date(item.timestamp).toLocaleString()}`;
      body.appendChild(time);

      node.appendChild(body);

      // Click opens full report
      node.addEventListener("click", () =>
        pTabsCreate({
          url: `${API_BASE}/get-report/${item.id}?token=${encodeURIComponent(jwt)}`
        })
      );

      historyList.appendChild(node);
    }); // ✅ closes forEach
  } catch (err) {
    console.error(err);
    historyList.innerHTML = "<div class='muted'>⚠️ Failed to load history.</div>";
  }
} // ✅ closes function

// --- LOGIN / LOGOUT ---
loginLink?.addEventListener("click", () =>
  pTabsCreate({ url: chrome.runtime.getURL("login.html") })
);

logoutBtn?.addEventListener("click", async () => {
  await new Promise((res) => chrome.storage.local.remove([KEY_JWT, KEY_USER], res));
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
  pTabsCreate({ url: "https://www.fakenewsdetector101.com/" });
});

// --- INIT ---
document.addEventListener("DOMContentLoaded", async () => {
  // ✅ Privacy Policy Check (only open once)
  const { fnd_privacyAccepted, fnd_privacyShown } = await pStorageGet([
    "fnd_privacyAccepted",
    "fnd_privacyShown"
  ]);

  if (!fnd_privacyAccepted) {
    if (!fnd_privacyShown) {
      await pStorageSet({ fnd_privacyShown: true });
      await pTabsCreate({ url: chrome.runtime.getURL("privacy_policy.html") });
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

  const o = await pStorageGet(KEY_LASTSCAN);
  const last = o?.[KEY_LASTSCAN];
  if (last) renderResults(last.url, last.headline, last);

  activate(tabScan);
  show(viewScan);
});
