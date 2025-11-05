// filename: login.js
// ===========================================================
// Fake News Detector — Login Page (Smooth Warning + Clean Mode Switch + Server Connect)
// Firefox-compatible (Promise wrappers)
// ===========================================================

const API_BASE = "https://fakenewsdetector-zjzs.onrender.com";
const msg = document.getElementById("msg");
const loader = document.getElementById("loader");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const loginBtn = document.getElementById("login-btn");
const registerBtn = document.getElementById("register-btn");
const switchMode = document.getElementById("switch-mode");
const backPopup = document.getElementById("back-popup");
const passwordHint = document.getElementById("password-hint");
const passwordWarning = document.getElementById("password-warning");

let isRegisterMode = false;

// ---- Promise helpers ----
const pStorageSet = (obj) => new Promise((res) => chrome.storage.local.set(obj, res));
const pTabsCreate  = (opt) => new Promise((res) => chrome.tabs.create(opt, res));

// ---- UI Helpers ----
function showMessage(text = "", type = "") {
  msg.textContent = text;
  msg.className = "msg " + type;
}

function toggleLoader(show) {
  loader.style.display = show ? "block" : "none";
  loginBtn.disabled = show;
  registerBtn.disabled = show;
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

// ---- Switch Between Login and Register ----
function switchFormMode() {
  isRegisterMode = !isRegisterMode;
  updatePasswordVisibility();

  if (isRegisterMode) {
    loginBtn.style.display = "none";
    registerBtn.style.display = "block";
    switchMode.textContent = "here to login";
    showMessage("Creating a new account.", "info");
  } else {
    loginBtn.style.display = "block";
    registerBtn.style.display = "none";
    switchMode.textContent = "here to register";
    showMessage("", "");
  }
}

// ---- Submit Handler ----
async function handleSubmit(isRegister = false) {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();

  if (!username || !password) {
    showMessage("Please fill in both fields.", "error");
    return;
  }

  if (isRegister && password.length < 8) {
    showMessage("Password must be at least 8 characters long.", "error");
    return;
  }

  toggleLoader(true);
  showMessage(isRegister ? "Registering..." : "Logging in...", "info");

  try {
    const endpoint = isRegister ? "/register" : "/login";
    const res = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Action failed");

    if (isRegister) {
      showMessage("✅ Registered successfully!", "success");
      setTimeout(async () => {
        await pTabsCreate({ url: chrome.runtime.getURL("success.html") });
        window.close();
      }, 1000);
      return;
    }

    // ✅ Login success
    await pStorageSet({ fnd_jwt: data.token, fnd_username: data.username });

    showMessage("✅ Login successful! Redirecting...", "success");
    backPopup.style.display = "none";

    setTimeout(async () => {
      await pTabsCreate({ url: chrome.runtime.getURL("welcome.html") });
      window.close();
    }, 1000);

  } catch (err) {
    if (String(err.message).includes("Failed to fetch")) {
      showMessage("🌐 Unable to connect to the server.", "error");
    } else {
      showMessage("⚠️ " + err.message, "error");
    }
  } finally {
    toggleLoader(false);
  }
}

// ---- Event Listeners ----
loginBtn.addEventListener("click", () => handleSubmit(false));
registerBtn.addEventListener("click", () => handleSubmit(true));
switchMode.addEventListener("click", (e) => { e.preventDefault(); switchFormMode(); });
backPopup.addEventListener("click", (e) => { e.preventDefault(); window.close(); });

// ---- Default State ----
registerBtn.style.display = "none";
passwordHint.style.display = "none";
passwordWarning.classList.remove("visible");
showMessage("", "");

// ---- Password Feedback (Only in Register Mode) ----
passwordInput.addEventListener("input", () => {
  if (!isRegisterMode) return;
  const pass = passwordInput.value;
  if (pass.length === 0)      passwordHint.style.color = "#555";
  else if (pass.length < 8)   passwordHint.style.color = "#e53935";
  else                        passwordHint.style.color = "#2e7d32";
});
