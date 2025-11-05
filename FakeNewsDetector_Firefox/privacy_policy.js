// filename: privacy_policy.js
// ===========================================================
// Fake News Detector — Privacy Policy Redirect Script (FINAL)
// Firefox-compatible
// ===========================================================

const pStorageSet = (obj) => new Promise((res) => chrome.storage.local.set(obj, res));
const pTabsCreate  = (opt) => new Promise((res) => chrome.tabs.create(opt, res));

document.addEventListener("DOMContentLoaded", () => {
  const agreeBtn = document.getElementById("agree-btn");
  if (!agreeBtn) return;

  agreeBtn.addEventListener("click", async () => {
    try {
      const mainURL = "https://www.fakenewsdetector101.com/fakenews.php#trusted-news";

      // ✅ Save acceptance flag BEFORE opening main site
      await pStorageSet({ fnd_privacyAccepted: true });

      // ✅ Open main website
      await pTabsCreate({ url: mainURL, active: true });

      // ✅ Close privacy policy tab
      window.close();

    } catch (err) {
      console.error("❌ Failed to open main website:", err);
      alert("Something went wrong. Please try again.");
    }
  });
});
