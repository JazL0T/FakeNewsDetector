// filename: welcome.js
// ===========================================================
// Fake News Detector — Welcome Page Script
// Open New Tab → Close Old Tabs → Jump to #trusted-news
// Firefox-compatible
// ===========================================================

const pTabsQuery  = (q)   => new Promise((res) => chrome.tabs.query(q, res));
const pTabsCreate = (opt) => new Promise((res) => chrome.tabs.create(opt, res));
const pTabRemove  = (id)  => new Promise((res) => chrome.tabs.remove(id, res));

document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("start-btn");
  if (!startBtn) return;

  startBtn.addEventListener("click", async () => {
    startBtn.disabled = true;
    startBtn.style.opacity = "0.8";
    startBtn.style.cursor = "wait";
    startBtn.innerHTML = `
      <span class="spinner"></span> Launching Fake News Detector 101...
    `;

    try {
      const mainDomain = "fakenewsdetector101.com";
      const targetURL = "https://www.fakenewsdetector101.com/fakenews.php#trusted-news";

      // ✅ Get all open tabs
      const tabs = await pTabsQuery({});

      // ✅ Close all existing Fake News Detector tabs
      const matchingTabs = tabs.filter(tab => tab.url && tab.url.includes(mainDomain));
      for (const tab of matchingTabs) {
        try { await pTabRemove(tab.id); } catch (e) { console.warn("Could not close tab:", e); }
      }

      // ✅ Open new tab to the target section
      await pTabsCreate({ url: targetURL, active: true });

      // ✅ Close the welcome page after short delay
      setTimeout(() => window.close(), 800);

    } catch (err) {
      console.error("⚠️ Failed to open main website:", err);
      alert("Please visit https://www.fakenewsdetector101.com/fakenews.php#trusted-news manually.");
    }
  });
});
