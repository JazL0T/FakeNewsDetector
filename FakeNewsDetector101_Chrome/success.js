// ===========================================================
// Fake News Detector — Success Page (Redirect to Login + CSP Safe)
// ===========================================================

document.addEventListener("DOMContentLoaded", () => {
  const loginRedirect = document.getElementById("login-redirect");

  loginRedirect.addEventListener("click", async () => {
    loginRedirect.textContent = "Redirecting...";
    loginRedirect.disabled = true;
    loginRedirect.style.opacity = "0.8";

    setTimeout(() => {
      // ✅ Go back to the login page
      window.location.href = "login.html";
    }, 700);
  });
});
