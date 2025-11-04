// ===========================================================
// Fake News Detector — Welcome Page Script (Return to Previous Website)
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {
  const startBtn = document.getElementById("start-btn");

  startBtn.addEventListener("click", async () => {
    startBtn.textContent = "Launching...";
    startBtn.disabled = true;
    startBtn.style.opacity = "0.8";

    try {
      // ✅ Find the last active tab (the user’s previous site)
      const [currentTab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const allTabs = await chrome.tabs.query({ currentWindow: true });

      // Get a different tab (not this welcome.html tab)
      const previousTab = allTabs.find(tab => tab.id !== currentTab.id);

      setTimeout(() => {
        if (previousTab) {
          // ✅ Focus the previous website tab
          chrome.tabs.update(previousTab.id, { active: true });
          // ✅ Close the welcome page
          chrome.tabs.remove(currentTab.id);
        } else {
          // Fallback — just close
          window.close();
        }
      }, 800);
    } catch (e) {
      console.warn("⚠️ Could not return to previous website:", e);
      window.close();
    }
  });
});
