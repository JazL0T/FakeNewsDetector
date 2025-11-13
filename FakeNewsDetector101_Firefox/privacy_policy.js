// ===========================================================
// Fake News Detector — Privacy Policy Redirect Script (FINAL)
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {
  const agreeBtn = document.getElementById("agree-btn");
  if (!agreeBtn) return;

  agreeBtn.addEventListener("click", async () => {
    try {
      const mainURL = "https://www.fakenewsdetector101.com/fakenews.php#trusted-news";

      
      await chrome.storage.local.set({ fnd_privacyAccepted: true });

      
      await chrome.tabs.create({ url: mainURL, active: true });

    
      window.close();

    } catch (err) {
      console.error("❌ Failed to open main website:", err);
      alert("Something went wrong. Please try again.");
    }
  });
});
