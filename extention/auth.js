// auth.js

document.addEventListener("DOMContentLoaded", () => {
  const loginEmail = document.getElementById("login-email");
  const loginPassword = document.getElementById("login-password");
  const loginBtn = document.getElementById("login-btn");
  const loginMsg = document.getElementById("login-msg");

  const regEmail = document.getElementById("reg-email");
  const regPassword = document.getElementById("reg-password");
  const registerBtn = document.getElementById("register-btn");
  const regMsg = document.getElementById("reg-msg");

  const toRegister = document.getElementById("to-register");
  const toLogin = document.getElementById("to-login");

  // 🌍 AUTO DETECT BASE URL (Localhost or Render)
  const BASE_URL =
    location.hostname === "localhost" || location.hostname === "127.0.0.1"
      ? "http://127.0.0.1:5000"
      : "https://fakenewsdetector101.onrender.com"; // <-- change this to your Render URL

  // ---------------- REGISTER ----------------
  if (registerBtn) {
    registerBtn.addEventListener("click", async () => {
      const email = regEmail.value.trim();
      const password = regPassword.value.trim();

      if (!email || !password) {
        regMsg.textContent = "⚠️ Please fill in all fields.";
        regMsg.style.color = "red";
        return;
      }

      regMsg.textContent = "⏳ Registering...";
      try {
        const res = await fetch(`${BASE_URL}/register`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        const data = await res.json();
        if (res.ok) {
          regMsg.textContent = "✅ Registration successful! Redirecting to login...";
          regMsg.style.color = "green";
          setTimeout(() => {
            window.location.href = "login.html";
          }, 1200);
        } else {
          regMsg.textContent = "❌ " + (data.error || "Registration failed.");
          regMsg.style.color = "red";
        }
      } catch (err) {
        console.error(err);
        regMsg.textContent = "❌ Server error — please try again.";
        regMsg.style.color = "red";
      }
    });
  }

  // ---------------- LOGIN ----------------
  if (loginBtn) {
    loginBtn.addEventListener("click", async () => {
      const email = loginEmail.value.trim();
      const password = loginPassword.value.trim();

      if (!email || !password) {
        loginMsg.textContent = "⚠️ Please fill in all fields.";
        loginMsg.style.color = "red";
        return;
      }

      loginMsg.textContent = "⏳ Logging in...";
      try {
        const res = await fetch(`${BASE_URL}/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email, password }),
        });

        const data = await res.json();
        if (res.ok) {
          loginMsg.textContent = "✅ Login successful!";
          loginMsg.style.color = "green";

          chrome.storage.local.set(
            { loggedIn: true, userEmail: email },
            () => {
              setTimeout(() => {
                window.close(); // Close auth window
              }, 1000);
            }
          );
        } else {
          loginMsg.textContent = "❌ " + (data.error || "Invalid credentials.");
          loginMsg.style.color = "red";
        }
      } catch (err) {
        console.error(err);
        loginMsg.textContent = "❌ Server error — please try again.";
        loginMsg.style.color = "red";
      }
    });
  }

  // ---------------- SWITCH BETWEEN LOGIN/REGISTER ----------------
  if (toRegister) {
    toRegister.addEventListener("click", () => {
      window.location.href = "register.html";
    });
  }

  if (toLogin) {
    toLogin.addEventListener("click", () => {
      window.location.href = "login.html";
    });
  }

  // ---------------- PASSWORD TOGGLE ----------------
  document.querySelectorAll(".toggle-password").forEach((icon) => {
    icon.addEventListener("click", () => {
      const targetId = icon.getAttribute("data-target");
      const input = document.getElementById(targetId);
      if (input) {
        input.type = input.type === "password" ? "text" : "password";
        icon.textContent = input.type === "password" ? "👁️" : "🙈";
      }
    });
  });
});
