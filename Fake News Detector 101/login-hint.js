// login_hint.js — show password hint + warning only in register mode
document.addEventListener("DOMContentLoaded", () => {
  const pwd = document.getElementById("password");
  const hint = document.getElementById("password-hint");
  const warning = document.getElementById("password-warning");
  const msg = document.getElementById("msg");
  const switchMode = document.getElementById("switch-mode");
  let isRegisterMode = false;

  // Show/hide hint + warning based on mode
  function updateVisibility() {
    const show = isRegisterMode ? "block" : "none";
    hint.style.display = show;
    warning.style.display = show;
  }

  // Toggle login/register mode
  switchMode.addEventListener("click", (e) => {
    e.preventDefault();
    isRegisterMode = !isRegisterMode;
    updateVisibility();

    if (isRegisterMode) {
      msg.textContent = "Creating a new account — password must be at least 8 characters.";
      msg.className = "msg info";
    } else {
      msg.textContent = "";
      msg.className = "msg";
    }
  });

  // Optional: live password feedback
  pwd.addEventListener("input", () => {
    if (!isRegisterMode) return;
    if (pwd.value.length === 0) {
      hint.style.color = "var(--muted)";
    } else if (pwd.value.length < 8) {
      hint.style.color = "var(--error)";
    } else {
      hint.style.color = "var(--success)";
    }
  });

  // Initialize in login mode
  updateVisibility();
});
