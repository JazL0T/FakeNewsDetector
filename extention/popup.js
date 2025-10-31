// popup.js
document.addEventListener("DOMContentLoaded", () => {
  const output = document.getElementById("output");
  const reportContainer = document.getElementById("report-btn-container");
  const loginLink = document.getElementById("login-link");
  const scanBtn = document.getElementById("scan-btn");

  // 🌍 AUTO DETECT BASE URL (Localhost or Render)
  const BASE_URL =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:5000"
      : "https://fakenewsdetector101.onrender.com"; // <-- change this to your actual Render URL

  // ---------------- Scan Button ----------------
  scanBtn.addEventListener("click", () => {
    output.textContent = "🔄 Scanning page...";
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs[0]?.id) {
        output.textContent = "❌ No active tab.";
        return;
      }

      chrome.tabs.sendMessage(tabs[0].id, { action: "scanPage" }, (response) => {
        if (!response) {
          output.textContent = "❌ No response from content script.";
          return;
        }
        if (response.error) {
          output.textContent = "❌ Error: " + response.error;
          return;
        }

        const result = response.result;
        chrome.storage.local.set({ latestPrediction: result }, () => {
          displayPrediction(result);
          saveToHistory(result);
        });
      });
    });
  });

  // ---------------- Load Stored Data ----------------
  chrome.storage.local.get(["latestPrediction", "loggedIn", "userEmail"], (data) => {
    const pred = data.latestPrediction;
    const loggedIn = !!data.loggedIn;
    const email = data.userEmail || "";

    if (pred) displayPrediction(pred);

    if (loggedIn) {
      createReportButtons(email);
    } else {
      showLoginButton();
    }
  });

  // ---------------- Helper: Save to History ----------------
  function saveToHistory(result) {
    chrome.storage.local.get(["scanHistory"], (data) => {
      const history = data.scanHistory || [];
      const newEntry = { ...result, timestamp: Date.now() };
      history.unshift(newEntry);
      chrome.storage.local.set({ scanHistory: history });
    });
  }

  // ---------------- Helper: Display Prediction ----------------
  function displayPrediction(pred) {
    const label = (pred.prediction || "").toLowerCase();
    const conf = (pred.confidence * 100).toFixed(1);
    const fakeProb = ((pred.class_probs?.["0"] ?? 0) * 100).toFixed(1);
    const realProb = ((pred.class_probs?.["1"] ?? 0) * 100).toFixed(1);
    const emoji = label.includes("fake") ? "⚠️ FAKE NEWS" : "✅ REAL NEWS";

    output.innerHTML = `
      <div style="padding:12px; border-radius:10px; background:#2c3e50; color:#fff; font-weight:600;">
        <div style="font-size:18px;">${emoji}</div>
        <div style="margin-top:6px;">Headline: ${pred.headline || "No headline"}</div>
        <div style="margin-top:6px;">Confidence: <b>${conf}%</b></div>
        <div style="margin-top:10px; height:18px; display:flex; overflow:hidden; border-radius:10px; background:#34495e;">
          <div style="flex:0 0 ${Math.max(fakeProb, 10)}%; background:#e74c3c; color:#fff; font-size:12px; display:flex; align-items:center; justify-content:center; border-radius:10px 0 0 10px;">
            Fake ${fakeProb}%
          </div>
          <div style="flex:0 0 ${Math.max(realProb, 10)}%; background:#3498db; color:#fff; font-size:12px; display:flex; align-items:center; justify-content:center; border-radius:0 10px 10px 0;">
            Real ${realProb}%
          </div>
        </div>
      </div>
    `;
  }

  // ---------------- Auth: Logged In ----------------
  function createReportButtons(email) {
    const reportBtn = document.createElement("button");
    reportBtn.textContent = "View Full Report";
    reportBtn.onclick = () => chrome.tabs.create({ url: `${BASE_URL}/full-report` });
    reportContainer.appendChild(reportBtn);

    const historyBtn = document.createElement("button");
    historyBtn.textContent = "View Scan History";
    historyBtn.onclick = showHistory;
    reportContainer.appendChild(historyBtn);

    loginLink.style.display = "block";
    loginLink.textContent = `Logout (${email})`;
    loginLink.onclick = () => {
      chrome.storage.local.set({ loggedIn: false, userEmail: "" }, () => {
        loginLink.style.display = "none";
        reportContainer.innerHTML = "";
        reportContainer.appendChild(scanBtn);
        output.innerHTML = "🔍 You have logged out.";
      });
    };
  }

  // ---------------- Auth: Not Logged In ----------------
  function showLoginButton() {
    loginLink.style.display = "block";
    loginLink.textContent = "Login / Register";
    loginLink.onclick = () => chrome.tabs.create({ url: chrome.runtime.getURL("login.html") });
  }

  // ---------------- History Page ----------------
  function showHistory() {
    output.innerHTML = "🔄 Loading history...";
    scanBtn.style.display = "none";
    reportContainer.innerHTML = "";

    const backBtn = document.createElement("button");
    backBtn.textContent = "⬅ Back to Main";
    backBtn.onclick = showMain;
    reportContainer.appendChild(backBtn);

    chrome.storage.local.get(["scanHistory"], (data) => {
      const history = data.scanHistory || [];
      if (history.length === 0) {
        output.innerHTML = "📄 No scan history available yet.";
        return;
      }

      output.innerHTML = "<h4>Scan History</h4>";
      history.forEach((scan, idx) => {
        const container = document.createElement("div");
        container.className = "history-card";
        container.style = `
          padding: 10px; margin-bottom: 6px; border-radius: 8px; background: #f0f2f5;
          border-left: 4px solid ${scan.prediction.toLowerCase() === "fake" ? "#e74c3c" : "#3498db"};
          cursor: pointer;
        `;
        const date = new Date(scan.timestamp).toLocaleString();
        container.innerHTML = `
          <div><b>#${idx + 1}:</b> ${scan.headline || "No headline"}</div>
          <div>Prediction: ${scan.prediction} | Confidence: ${(scan.confidence * 100).toFixed(1)}%</div>
          <div style="font-size:12px; color:#555;">${date}</div>
        `;
        container.addEventListener("click", () => displayFullReport(scan));
        output.appendChild(container);
      });
    });
  }

  // ---------------- Back to Main ----------------
  function showMain() {
    output.innerHTML = "🔍 Waiting for scan...";
    reportContainer.innerHTML = "";
    scanBtn.style.display = "block";
    reportContainer.appendChild(scanBtn);

    chrome.storage.local.get(["loggedIn", "userEmail"], (data) => {
      if (data.loggedIn) createReportButtons(data.userEmail);
      else showLoginButton();
    });
  }

  // ---------------- Full Report ----------------
  function displayFullReport(scan) {
    output.innerHTML = "";
    reportContainer.innerHTML = "";

    const backBtn = document.createElement("button");
    backBtn.textContent = "⬅ Back to History";
    backBtn.onclick = showHistory;
    reportContainer.appendChild(backBtn);

    const label = scan.prediction.toLowerCase();
    const emoji = label.includes("fake") ? "⚠️ FAKE NEWS" : "✅ REAL NEWS";
    const conf = (scan.confidence * 100).toFixed(1);

    output.innerHTML = `
      <div style="padding:12px; border-radius:10px; background:#2c3e50; color:#fff;">
        <div style="font-size:18px;">${emoji}</div>
        <div>Confidence: <b>${conf}%</b></div>
        <div>Headline: ${scan.headline || "No headline detected"}</div>
        <div>URL: <a href="${scan.url}" target="_blank" style="color:#61dafb;">${scan.url}</a></div>
      </div>
      <pre style="margin-top:10px; padding:10px; background:#f0f2f5; border-radius:8px; max-height:180px; overflow-y:auto;">
${scan.text || "No article text captured."}
      </pre>
      <button style="margin-top:10px;" id="copyText">📋 Copy Article Text</button>
    `;

    document.getElementById("copyText").onclick = () =>
      navigator.clipboard.writeText(scan.text || "").then(() => alert("Article text copied!"));
  }
});
