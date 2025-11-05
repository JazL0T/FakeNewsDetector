// filename: background.js
// ===========================================================
// Fake News Detector 101 — Show Privacy Policy Only Once (FF-safe)
// ===========================================================

const pStorageGet = (keys) => new Promise((resolve) => chrome.storage.local.get(keys, resolve));
const pTabsCreate  = (opts) => new Promise((resolve) => chrome.tabs.create(opts, resolve));

chrome.runtime.onInstalled.addListener(async (details) => {
  try {
    const { fnd_privacyAccepted } = await pStorageGet("fnd_privacyAccepted");

    // ✅ Show only if new install and user hasn't accepted yet
    if (details.reason === "install" && !fnd_privacyAccepted) {
      console.log("📜 Showing privacy policy for first-time user");
      await pTabsCreate({ url: chrome.runtime.getURL("privacy_policy.html") });
    } else {
      console.log("✅ Privacy already accepted, skipping page");
    }

  } catch (err) {
    console.error("⚠️ Privacy check failed:", err);
  }
});
