chrome.runtime.onInstalled.addListener(async () => {
  const { fnd_privacyAccepted } = await chrome.storage.local.get("fnd_privacyAccepted");

  if (!fnd_privacyAccepted) {
    await chrome.storage.local.set({ fnd_privacyShown: true });
    chrome.tabs.create({ url: chrome.runtime.getURL("privacy_policy.html") });
  }
});
