// ===========================================================
// Fake News Detector — Popup (Cross-browser MV2-compatible patch)
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
const loginLink = document.getElementById("login-link"); // <--- THIS ONE IS ESSENTIAL
const logoutBtn = document.getElementById("logout-btn");
const websiteBtn = document.getElementById("visit-website");

// 🆕 For language & model used display
let resLanguageEl = null;
let resModelEl = null;

// cross-browser extension API (browser in Firefox, chrome in Chrome)
const ext = typeof browser !== "undefined" ? browser : chrome;

// --- Cross-browser storage helpers ---
function storageGet(keys) {
  return new Promise((resolve) => {
    try {
      const rv = ext.storage.local.get(keys, (res) => {
        // If callback-style, use callback result
        if (typeof res !== "undefined") return resolve(res);
        resolve({});
      });
      // If it returned a promise (browser.*), use it
      if (rv && typeof rv.then === "function") {
        rv.then(resolve).catch(() => resolve({}));
      }
    } catch (e) {
      resolve({});
    }
  });
}

function storageSet(obj) {
  return new Promise((resolve) => {
    try {
      const rv = ext.storage.local.set(obj, () => resolve());
      if (rv && typeof rv.then === "function") rv.then(resolve).catch(resolve);
    } catch (e) {
      resolve();
    }
  });
}

function storageRemove(keys) {
  return new Promise((resolve) => {
    try {
      const rv = ext.storage.local.remove(keys, () => resolve());
      if (rv && typeof rv.then === "function") rv.then(resolve).catch(resolve);
    } catch (e) {
      resolve();
    }
  });
}

// --- Cross-browser tabs.executeScript / scripting.executeScript wrapper ---
function executeScriptCompat(tabId, details) {
  return new Promise((resolve) => {
    try {
      if (ext.scripting && typeof ext.scripting.executeScript === "function") {
        // new API: ext.scripting.executeScript({ target: { tabId }, files: [...] })
        const rv = ext.scripting.executeScript(Object.assign({}, details, { target: { tabId } }));
        if (rv && typeof rv.then === "function") return rv.then(resolve).catch(() => resolve(null));
        // fallback to callback signature (rare)
        return resolve(null);
      } else if (ext.tabs && typeof ext.tabs.executeScript === "function") {
        // MV2 style: ext.tabs.executeScript(tabId, { file: "content.js" }, callback)
        const file = (details.files && details.files[0]) || details.code;
        const cb = (res) => resolve(res);
        try {
          // Some browsers accept an object {file: ...} (Chrome MV2)
          ext.tabs.executeScript(tabId, details, cb);
        } catch (e) {
          // fallback: try with file only
          ext.tabs.executeScript(tabId, { file }, cb);
        }
      } else {
        resolve(null);
      }
    } catch (e) {
      resolve(null);
    }
  });
}

// --- Cross-browser sendMessage to a tab ---
function sendMessageToTab(tabId, message) {
  return new Promise((resolve) => {
    try {
      const rv = ext.tabs.sendMessage(tabId, message, (res) => {
        if (typeof res !== "undefined") return resolve(res);
        // if there was a runtime error or no response, resolve null
        resolve(null);
      });
      if (rv && typeof rv.then === "function") {
        rv.then(resolve).catch(() => resolve(null));
      }
    } catch (e) {
      resolve(null);
    }
  });
}

// --- Helpers ---
function showStatus(msg, isError = false) {
  if (scanStatus) {
    scanStatus.textContent = msg;
    scanStatus.style.color = isError ? "#e53935" : "#1565c0";
  }
}

function toggleLoading(isLoading) {
  if (spinner) spinner.classList.toggle("hidden", !isLoading);
  if (scanBtn) {
    scanBtn.disabled = isLoading;
    scanBtn.textContent = isLoading ? "⏳ Scanning..." : "Scan this page";
  }
}

function fmtPct(n) {
  const num = Number(n);
  return Number.isNaN(num) ? "0%" : (num * 100).toFixed(1) + "%";
}

async function getAuth() {
  const v = await storageGet([KEY_JWT, KEY_USER]);
  return { jwt: v[KEY_JWT], username: v[KEY_USER] };
}

function fadeIn(el) {
  if (!el) return;
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
  if (summary) summary.classList.add("hidden");
  if (resCard) resCard.classList.add("hidden");
  if (fullReportBtn) fullReportBtn.classList.add("hidden");
  showStatus("Extracting article...");

  // query active tab
  let tab = null;
  try {
    const tabsRv = await new Promise((resolve) => {
      try {
        const rv = ext.tabs.query({ active: true, currentWindow: true }, (res) => resolve(res));
        if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve([]));
      } catch (e) {
        resolve([]);
      }
    });
    tab = Array.isArray(tabsRv) ? tabsRv[0] : tabsRv && tabsRv[0];
  } catch (e) {
    tab = null;
  }

  if (!tab || !tab.id) {
    toggleLoading(false);
    return showStatus("No active tab found.", true);
  }

  try {
    // try to inject content.js (if needed)
    await executeScriptCompat(tab.id, { files: ["content.js"] });
  } catch (err) {
    console.warn("content.js injection skipped", err);
  }

  try {
    const result = await sendMessageToTab(tab.id, { action: "scanPage" });
    if (!result) throw new Error("Could not connect to content script.");
    if (result.error) throw new Error(result.error);

    const { headline, text, url } = result.result || result;
    if (!text) throw new Error("No article text found.");

    // Cached scan logic
    const stored = await storageGet([KEY_LASTSCAN]);
    const last = stored[KEY_LASTSCAN];

    if (last && last.url === url) {
      renderResults(url, headline, last);
      showStatus("✅ Same article detected — showing cached result.");

      if (jwt && username) {
        try {
          const headers = {
            "Content-Type": "application/json",
            Authorization: `Bearer ${jwt}`,
          };
          await fetch(`${API_BASE}/predict`, {
            method: "POST",
            headers,
            body: JSON.stringify({ headline, text, url }),
          });
          console.log("Cached scan synced to backend for history.");
          showStatus("Cached scan saved to your account history!");
          fullReportBtn.classList.remove("hidden");
        } catch (syncErr) {
          console.warn("⚠️ Failed to sync cached scan:", syncErr);
          showStatus("⚠️ Failed to sync cached scan — try rescanning.", true);
        }
      }

      toggleLoading(false);
      return;
    }

    // Normal scanning path
    const headers = { "Content-Type": "application/json" };
    if (jwt) headers["Authorization"] = `Bearer ${jwt}`;

    showStatus("Analyzing content...");
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers,
      body: JSON.stringify({ headline, text, url }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Prediction failed.");

    renderResults(url, headline, data);
    await storageSet({ [KEY_LASTSCAN]: data });

    if (username && jwt) fullReportBtn.classList.remove("hidden");
    showStatus("");
  } catch (err) {
    console.error(err);
    showStatus(`⚠️ ${err.message}`, true);
  } finally {
    toggleLoading(false);
  }
}

// --- RESULTS ---
function renderResults(url, headline, data) {
  if (resUrl) resUrl.innerHTML = url ? `<a href="${url}" target="_blank" class="result-link">${url}</a>` : "—";
  if (resHeadline) resHeadline.textContent = headline || "—";

  const pred = data.prediction;
  if (resPrediction) {
    resPrediction.textContent = pred;
    resPrediction.className =
      "result-badge " +
      (pred === "Fake"
        ? "fake"
        : pred === "Likely Real"
        ? "uncertain"
        : pred === "Real"
        ? "real"
        : "uncertain");
  }

  const conf = fmtPct(data.confidence);
  if (resConfidence) resConfidence.textContent = conf;

  const bar = document.getElementById("confidence-fill");
  if (bar) {
    bar.style.width = conf;
    const colorMap = {
      "Fake": "#e53935",
      "Likely Real": "#f9a825",
      "Real": "#2e7d32",
    };
    bar.style.backgroundColor = colorMap[pred] || "#f9a825";
  }

  const t = data.trustability || {};
  if (resDomain) resDomain.textContent = t.domain || "—";
  if (resTrustScore) resTrustScore.textContent = t.trust_score ?? "—";
  if (resTrustCat) {
    resTrustCat.textContent = t.category || "—";
    resTrustCat.className =
      "result-badge " +
      (t.category === "Trusted" || t.category === "Trusted (Malaysia)"
        ? "real"
        : t.category === "Suspicious"
        ? "fake"
        : "uncertain");
  }

  if (!resLanguageEl || !resModelEl) {
    resLanguageEl = document.createElement("p");
    resModelEl = document.createElement("p");
    resLanguageEl.className = "packed-line";
    resModelEl.className = "packed-line";
    resCard.querySelector(".result-grid")?.appendChild(resLanguageEl);
    resCard.querySelector(".result-grid")?.appendChild(resModelEl);
  }

  resLanguageEl.innerHTML = `<b>Language Detected:</b> ${data.language || "Unknown"}`;
  resModelEl.innerHTML = `<b>Model Used:</b> ${data.model_used || "N/A"}`;

  // Smart Summary (unchanged)
  let summaryMsg = "";
  let summaryColor = "#e3f2fd";

  if (pred === "Likely Real") {
    summaryMsg = `🟡 This article appears real based on writing style and trusted source (${t.domain}), but minor inconsistencies exist.`;
    summaryColor = "#fff8e1";
  } else if (pred === "Fake" && t.category === "Trusted") {
    summaryMsg = `🟠 The content looks suspicious, but the source (${t.domain}) is reputable. Possibly satire or opinion-based.`;
    summaryColor = "#fff3cd";
  } else if (pred === "Fake" && t.category === "Suspicious") {
    summaryMsg = `🔴 Both the article and its source (${t.domain || "unknown"}) seem unreliable. Proceed with caution.`;
    summaryColor = "#ffebee";
  } else if (pred === "Real" && (t.category || "").startsWith("Trusted")) {
    summaryMsg = `🟢 Legitimate article from a verified, trusted outlet (${t.domain}).`;
    summaryColor = "#e8f5e9";
  } else if (pred === "Real" && t.category === "Uncertain") {
    summaryMsg = `🟡 The writing seems factual, but the source (${t.domain}) is unverified. Check other outlets.`;
    summaryColor = "#fffde7";
  } else {
    summaryMsg = `⚪ Mixed indicators — verify before trusting completely.`;
    summaryColor = "#f5f5f5";
  }

  if (summary) {
    summary.style.backgroundColor = summaryColor;
    summaryText.textContent = summaryMsg;
    fadeIn(summary);
  }
  if (resCard) fadeIn(resCard);
}

// --- FULL REPORT ---
fullReportBtn?.addEventListener("click", async () => {
  const { jwt } = await getAuth();
  if (!jwt) return showStatus("🔒 Please log in to view reports.", true);

  try {
    const res = await fetch(`${API_BASE}/get-history`, {
      headers: { Authorization: `Bearer ${jwt}` }
    });
    const data = await res.json();
    if (!res.ok || !data.history?.length)
      return showStatus("⚠️ No reports found.", true);

    const latest = data.history[0];
    const reportUrl = `${API_BASE}/get-report/${latest.id}?token=${encodeURIComponent(jwt)}`;

    // Open in same tab using tabs.update (cross-browser)
    try {
      const tabsRv = await new Promise((resolve) => {
        try {
          const rv = ext.tabs.query({ active: true, currentWindow: true }, (res) => resolve(res));
          if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve([]));
        } catch (e) {
          resolve([]);
        }
      });
      const tab = Array.isArray(tabsRv) ? tabsRv[0] : tabsRv && tabsRv[0];
      if (tab && tab.id && ext.tabs.update) {
        await new Promise((resolve) => {
          try {
            const rv = ext.tabs.update(tab.id, { url: reportUrl }, resolve);
            if (rv && typeof rv.then === "function") rv.then(resolve).catch(resolve);
          } catch (e) {
            resolve();
          }
        });
      } else {
        window.open(reportUrl, "_self");
      }
    } catch (e) {
      window.open(reportUrl, "_self");
    }

  } catch (err) {
    console.error(err);
    showStatus("⚠️ Failed to open full report.", true);
  }
});

// --- HISTORY (Backend-synced + Color Consistent) ---
async function renderHistory() {
  const { jwt, username } = await getAuth();

  if (!jwt || !username) {
    if (historyList) historyList.innerHTML = "";
    if (historyHint) {
      historyHint.style.display = "block";
      historyHint.textContent = "🔒 Log in to view your past scans.";
    }
    return;
  }

  if (historyHint) {
    historyHint.style.display = "none";
    historyList.innerHTML = "<div class='muted'>Loading your history...</div>";
  }

  try {
    const res = await fetch(`${API_BASE}/get-history`, {
      headers: { Authorization: `Bearer ${jwt}` },
    });
    const data = await res.json();

    if (!res.ok || !data.history?.length) {
      historyList.innerHTML =
        "<div class='muted'>No scans yet. Start scanning to build your history!</div>";
      return;
    }

    historyList.innerHTML = "";

    data.history.slice(0, 10).forEach((item) => {
      const trust = item.trustability || {};
      const cat = (trust.category || "").toLowerCase();

      const pred = (item.prediction || "").toLowerCase();
      let predLabel = "Uncertain";
      let predColor = "#f9a825";
      let predEmoji = "⚪";

      if (pred.includes("fake")) {
        predLabel = "Fake";
        predColor = "#e53935";
        predEmoji = "🔴";
      } else if (pred.includes("likely")) {
        predLabel = "Likely Real";
        predColor = "#f9a825";
        predEmoji = "🟡";
      } else if (pred.includes("real")) {
        predLabel = "Real";
        predColor = "#2e7d32";
        predEmoji = "🟢";
      }

      const predBadge = `
        <span style="
          background:${predColor}15;
          color:${predColor};
          font-weight:600;
          padding:3px 8px;
          border-radius:6px;
          font-size:13px;
        ">${predEmoji} ${predLabel}</span>`;

      const catColor =
        cat.includes("trusted") || cat.includes("malaysia")
          ? "#2e7d32"
          : cat.includes("suspicious")
          ? "#e53935"
          : "#f9a825";

      const node = document.createElement("div");
      node.className = "hist-card";
      node.innerHTML = `
        <div class="hist-header">${item.headline || "(No headline)"}</div>
        <div class="hist-body">
          <div class="hist-info">
            ${predBadge} <span style="color:#555;">(${fmtPct(item.confidence)})</span> —
            <span style="color:${catColor};font-weight:600;">${trust.category || "Unverified"}</span>
          </div>
          <div class="hist-time">🕒 ${new Date(item.timestamp).toLocaleString()}</div>
        </div>
      `;

      node.addEventListener("click", () =>
        ext.tabs.create({
          url: `${API_BASE}/get-report/${item.id}?token=${encodeURIComponent(jwt)}`,
        })
      );

      historyList.appendChild(node);
    });
  } catch (err) {
    console.error(err);
    historyList.innerHTML =
      "<div class='muted'>⚠️ Failed to load history.</div>";
  }
}

// --- LOGIN / LOGOUT ---
loginLink?.addEventListener("click", () =>
  ext.tabs.create({ url: ext.runtime.getURL("login.html") })
);

logoutBtn?.addEventListener("click", async () => {
  await storageRemove([KEY_JWT, KEY_USER]);
  if (userStatus) userStatus.textContent = "🔒 Guest mode";
  if (loginTip) loginTip.style.display = "block";
  if (logoutBtn) logoutBtn.classList.add("hidden");
  if (loginLink) loginLink.classList.remove("hidden");
  if (fullReportBtn) fullReportBtn.classList.add("hidden");
  if (historyList) historyList.innerHTML = "";
  if (historyHint) {
    historyHint.style.display = "block";
    historyHint.textContent = "🔒 Log in to view your past scans.";
  }
});

// --- WEBSITE BUTTON ---
websiteBtn?.addEventListener("click", () => {
  ext.tabs.create({ url: "https://www.fakenewsdetector101.com/" });
});

// --- INIT ---
document.addEventListener("DOMContentLoaded", async () => {
  const stored = await storageGet(["fnd_privacyAccepted", "fnd_privacyShown"]);
  const fnd_privacyAccepted = stored.fnd_privacyAccepted;
  const fnd_privacyShown = stored.fnd_privacyShown;

  if (!fnd_privacyAccepted) {
    if (!fnd_privacyShown) {
      await storageSet({ fnd_privacyShown: true });
      try { await ext.tabs.create({ url: ext.runtime.getURL("privacy_policy.html") }); } catch(e){}
    }
    window.close();
    return;
  }

  const { jwt, username } = await getAuth();
  const loggedIn = jwt && username;

  if (userStatus) userStatus.textContent = loggedIn ? `👤 Logged in as ${username}` : "🔓 Guest mode";
  if (loginTip) loginTip.style.display = loggedIn ? "none" : "block";

  if (!loggedIn) {
    if (loginLink) loginLink.classList.remove("hidden");
    if (logoutBtn) logoutBtn.classList.add("hidden");
  } else {
    if (loginLink) loginLink.classList.add("hidden");
    if (logoutBtn) logoutBtn.classList.remove("hidden");
  }

  const lastStored = await storageGet([KEY_LASTSCAN]);
  const last = lastStored[KEY_LASTSCAN];
  if (last) {
    renderResults(last.url, last.headline, last);
    if (loggedIn && fullReportBtn) fullReportBtn.classList.remove("hidden");
  }

  activate(tabScan);
  show(viewScan);
});
