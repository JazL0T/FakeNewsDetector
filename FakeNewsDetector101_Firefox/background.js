// ===============================================
// Fake News Detector 101 — Cross-Browser Background (MV2 Compatible)
// ===============================================

// Use the correct API in Firefox (browser.*) but fallback for Chrome
const ext = typeof browser !== "undefined" ? browser : chrome;

ext.runtime.onInstalled.addListener(async (details) => {
  try {
    // Use storage helper pattern that handles both callback and promise styles
    function storageGet(keys) {
      return new Promise((resolve) => {
        try {
          const rv = ext.storage.local.get(keys, (res) => {
            if (typeof res !== "undefined") return resolve(res);
            resolve({});
          });
          if (rv && typeof rv.then === "function") rv.then(resolve).catch(() => resolve({}));
        } catch (e) {
          resolve({});
        }
      });
    }

    const data = await storageGet("fnd_privacyAccepted");
    const fnd_privacyAccepted = data?.fnd_privacyAccepted;

    if (details.reason === "install" && !fnd_privacyAccepted) {
      console.log("📜 Showing privacy policy for first-time user");
      try {
        await ext.tabs.create({
          url: ext.runtime.getURL("privacy_policy.html")
        });
      } catch (e) {
        console.warn("Failed to open privacy tab:", e);
      }
    } else {
      console.log("✅ Privacy already accepted, skipping page");
    }

  } catch (err) {
    console.error("⚠️ Privacy check failed in background:", err);
  }
});
