// ===========================================================
// Fake News Detector — Welcome Page Script (Cross-browser fix)
// Open New Tab → Close Old Tabs → Jump to #trusted-news
// ===========================================================

const ext = typeof browser !== "undefined" ? browser : chrome;

function tabsQueryAll() {
  return new Promise((resolve) => {
    try {
      const rv = ext.tabs.query({}, (res) => resolve(res || []));
      if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve([]));
    } catch (e) {
      resolve([]);
    }
  });
}

function tabsRemove(ids) {
  return new Promise((resolve) => {
    try {
      const rv = ext.tabs.remove(ids, () => resolve());
      if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve());
    } catch (e) {
      resolve();
    }
  });
}

function tabsCreate(createProps) {
  return new Promise((resolve) => {
    try {
      const rv = ext.tabs.create(createProps, (tab) => resolve(tab));
      if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve(null));
    } catch (e) {
      resolve(null);
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("start-btn");
  if (!startBtn) return;

  startBtn.addEventListener("click", async () => {
    try {
      startBtn.disabled = true;
      startBtn.style.opacity = "0.8";
      startBtn.style.cursor = "wait";
      startBtn.innerHTML = `
        <span class="spinner"></span> Launching Fake News Detector 101...
      `;

      const mainDomain = "fakenewsdetector101.com";
      const targetURL = "https://www.fakenewsdetector101.com/fakenews.php#trusted-news";

      // Get all open tabs
      const tabs = await tabsQueryAll();

      // Close any existing tabs that point to our site to avoid duplicates
      const matching = (tabs || []).filter(t => t && t.url && t.url.includes(mainDomain));
      const idsToClose = matching.map(t => t.id).filter(Boolean);
      if (idsToClose.length) {
        try {
          await tabsRemove(idsToClose);
        } catch (e) { /* ignore */ }
      }

      // Open new tab to the target section
      await tabsCreate({ url: targetURL, active: true });

      // Close the welcome page after short delay (best-effort)
      setTimeout(() => {
        try { window.close(); } catch(e) {}
      }, 600);

    } catch (err) {
      console.error("⚠️ Failed to open main website:", err);
      alert("Please visit https://www.fakenewsdetector101.com/fakenews.php#trusted-news manually.");
      startBtn.disabled = false;
      startBtn.style.opacity = "1";
      startBtn.style.cursor = "pointer";
      startBtn.textContent = "Start Scanning";
    }
  });
});
