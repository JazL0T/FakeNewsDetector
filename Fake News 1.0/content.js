function extractHeadline() {
  const selectors = [
    'h1',
    'header h1',
    'article h1',
    'main h1',
    '[itemprop="headline"]',
    '.headline',
    '.article-title'
  ];
  for (const sel of selectors) {
    const el = document.querySelector(sel);
    if (el && el.innerText.trim()) return el.innerText.trim();
  }
  return document.title || "";
}

function extractArticleText() {
  const candidates = ['article', 'main', '.article', '.story', '.post-content', '.entry-content'];
  let best = "";
  candidates.forEach(sel => {
    document.querySelectorAll(sel).forEach(node => {
      const text = node.innerText?.trim() || "";
      if (text.length > best.length) best = text;
    });
  });
  if (best.length < 400) {
    const ps = Array.from(document.querySelectorAll('p'));
    const text = ps.map(p => p.innerText.trim()).join('\n');
    if (text.length > best.length) best = text;
  }
  return best;
}

function extractUrl() {
  const link = document.querySelector('link[rel="canonical"]');
  return (link && link.href) ? link.href : location.href;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.action === "scanPage") {
    try {
      const headline = extractHeadline();
      const text = extractArticleText();
      const url = extractUrl();
      if (!text || text.length < 100) {
        sendResponse({ error: "Could not extract enough article text from this page." });
        return true;
      }
      sendResponse({ result: { headline, text, url } });
    } catch (e) {
      sendResponse({ error: String(e) });
    }
    return true;
  }
});
