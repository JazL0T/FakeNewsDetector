document.addEventListener("DOMContentLoaded", () => {
  const agreeBtn = document.getElementById("agree-btn");

  if (!agreeBtn) return;

  agreeBtn.addEventListener("click", async () => {
    try {
      // Save acceptance flag and clear "shown" state
      await chrome.storage.local.set({
        fnd_privacyAccepted: true,
        fnd_privacyShown: false
      });

      // Close the policy tab
      chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
        if (tabs && tabs[0]) chrome.tabs.remove(tabs[0].id);
        else window.close();
      });
    } catch (err) {
      console.error("❌ Failed to accept privacy policy:", err);
      alert("Something went wrong. Please try again.");
    }
  });
});
