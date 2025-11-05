// compat.js — lightweight MV2/MV3 compatibility shim for Firefox

window.browser = window.browser || window.chrome;

// Provide a basic scripting shim for Chrome->Firefox compatibility
if (!chrome.scripting && chrome.tabs && !chrome.tabs.executeScript._shimAdded) {
  chrome.scripting = {
    executeScript: ({ target, files, func, args }) => {
      if (files && files.length) {
        return Promise.all(
          files.map(f => new Promise((resolve, reject) => {
            chrome.tabs.executeScript(target.tabId, { file: f }, () => {
              if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
              else resolve();
            });
          }))
        );
      }
      if (typeof func === 'function') {
        const code = `(${func}).apply(null, ${JSON.stringify(args || [])});`;
        return new Promise((resolve, reject) => {
          chrome.tabs.executeScript(target.tabId, { code }, () => {
            if (chrome.runtime.lastError) reject(chrome.runtime.lastError);
            else resolve();
          });
        });
      }
      return Promise.resolve();
    }
  };
  chrome.tabs.executeScript._shimAdded = true;
}
