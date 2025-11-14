// ===========================================================
// Fake News Detector — Login Page (Final Stable + Safe DOM + Success & Welcome Redirects)
// ===========================================================

const API_BASE = "https://fakenewsdetector-zjzs.onrender.com";
const msg = document.getElementById("msg");
const loader = document.getElementById("loader");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginBtn = document.getElementById("login-btn");
const switchMode = document.getElementById("switch-mode");
const backPopup = document.getElementById("back-popup");
const passwordHint = document.getElementById("password-hint");
const passwordWarning = document.getElementById("password-warning");

// Optional — only exist in popup.html
const userStatus = document.getElementById("user-status");
const loginTip = document.getElementById("login-tip");

let isRegisterMode = false;

// ===========================================================
// 🧩 Helpers
// ===========================================================
function showMessage(text = "", type = "") {
  msg.textContent = text;
  msg.className = "msg " + type;
}

function toggleLoader(show) {
  loader.style.display = show ? "block" : "none";
  loginBtn.disabled = show;
}

function safeSetText(el, text) {
  if (el && typeof el.textContent !== "undefined") el.textContent = text;
}

function safeDisplay(el, visible) {
  if (el && typeof el.style !== "undefined") {
    el.style.display = visible ? "block" : "none";
  }
}

function updatePasswordVisibility() {
  if (isRegisterMode) {
    passwordHint.style.display = "block";
    passwordWarning.classList.add("visible");
  } else {
    passwordHint.style.display = "none";
    passwordWarning.classList.remove("visible");
  }
}

// ===========================================================
// 🔄 Switch Between Login / Register
// ===========================================================
function switchFormMode() {
  isRegisterMode = !isRegisterMode;
  updatePasswordVisibility();

  if (isRegisterMode) {
    loginBtn.textContent = "Register";
    switchMode.textContent = "here to login";
    showMessage("Creating a new account.", "info");
  } else {
    loginBtn.textContent = "Login";
    switchMode.textContent = "here to register";
    showMessage("", "");
  }
}

// ===========================================================
// 🚀 Submit Handler (Login / Register)
// ===========================================================
async function handleSubmit() {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();

  if (!username || !password) {
    showMessage("⚠️ Please fill in both fields.", "error");
    return;
  }

  if (isRegisterMode && password.length < 8) {
    showMessage("⚠️ Password must be at least 8 characters long.", "error");
    return;
  }

  toggleLoader(true);
  showMessage(isRegisterMode ? "Registering..." : "Logging in...", "info");

  try {
    const endpoint = isRegisterMode ? "/register" : "/login";
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Action failed");

    // ✅ Registration Success → Redirect to success.html
    if (isRegisterMode) {
      showMessage("✅ Registration successful! Redirecting...", "success");
      setTimeout(() => {
        chrome.tabs.create({ url: chrome.runtime.getURL("success.html") });
        window.close();
      }, 1200);
      return;
    }

    // ✅ Login Success → Store user + Redirect to welcome.html
    await chrome.storage.local.set({
      fnd_jwt: data.token,
      fnd_username: data.username,
    });

    showMessage("✅ Login successful! Redirecting...", "success");

    safeSetText(userStatus, `👤 Logged in as ${data.username}`);
    safeDisplay(loginTip, false);

    setTimeout(() => {
      chrome.tabs.create({ url: chrome.runtime.getURL("welcome.html") });
      window.close();
    }, 1200);

  } catch (err) {
    if (err.message.includes("Failed to fetch")) {
      showMessage("🌐 Unable to connect to the server.", "error");
    } else {
      showMessage("⚠️ " + err.message, "error");
    }
  } finally {
    toggleLoader(false);
  }
}

// ===========================================================
// 🎛 Event Listeners
// ===========================================================
loginBtn.addEventListener("click", () => handleSubmit());
switchMode.addEventListener("click", (e) => {
  e.preventDefault();
  switchFormMode();
});
backPopup.addEventListener("click", (e) => {
  e.preventDefault();
  window.close();
});

// ===========================================================
// ⚙️ Default State
// ===========================================================
passwordHint.style.display = "none";
passwordWarning.classList.remove("visible");
showMessage("", "");

// ===========================================================
// 🔡 Live Password Strength (Register Mode)
// ===========================================================
passwordInput.addEventListener("input", () => {
  if (!isRegisterMode) return;
  const len = passwordInput.value.length;
  passwordHint.style.color =
    len === 0 ? "#555" : len < 8 ? "#e53935" : "#2e7d32";
});

// ===========================================================
// 🔁 Smooth Popup Reload Handler
// ===========================================================
chrome.runtime.onMessage.addListener((req) => {
  if (req.action === "reloadPopup") {
    console.log("🔁 Reloading popup after successful login...");
    window.location.reload();
  }
});
