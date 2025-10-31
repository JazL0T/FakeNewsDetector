document.addEventListener("DOMContentLoaded", () => {
  // Determine page
  const isLoginPage = document.getElementById("login-btn") !== null;
  const isRegisterPage = document.getElementById("register-btn") !== null;

  // ---------------- Login ----------------
  if(isLoginPage){
    const loginBtn = document.getElementById("login-btn");
    const emailInput = document.getElementById("login-email");
    const passwordInput = document.getElementById("login-password");
    const msgDiv = document.getElementById("login-msg");
    const switchLink = document.getElementById("to-register");

    loginBtn.addEventListener("click", async ()=>{
      const email = emailInput.value;
      const password = passwordInput.value;
      if(!email || !password){ msgDiv.textContent="Please fill all fields"; return; }

      try{
        const res = await fetch("http://127.0.0.1:5000/login", {
          method:"POST",
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({email,password})
        });
        const data = await res.json();
        if(data.success){
          chrome.storage.local.set({loggedIn:true,userEmail:email}, ()=>{
            msgDiv.textContent="Login successful!";
            setTimeout(()=>window.close(),1000); // close popup
          });
        } else msgDiv.textContent=data.message || "Login failed";
      } catch(e){ msgDiv.textContent="Error: "+e.message; }
    });

    switchLink.addEventListener("click", ()=>{
      window.location.href = chrome.runtime.getURL("register.html");
    });
  }

  // ---------------- Register ----------------
  if(isRegisterPage){
    const registerBtn = document.getElementById("register-btn");
    const emailInput = document.getElementById("reg-email");
    const passwordInput = document.getElementById("reg-password");
    const msgDiv = document.getElementById("reg-msg");
    const switchLink = document.getElementById("to-login");

    registerBtn.addEventListener("click", async ()=>{
      const email = emailInput.value;
      const password = passwordInput.value;
      if(!email || !password){ msgDiv.textContent="Please fill all fields"; return; }

      try{
        const res = await fetch("http://127.0.0.1:5000/register", {
          method:"POST",
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({email,password})
        });
        const data = await res.json();
        if(data.success){
          msgDiv.textContent="Registration successful! Redirecting to login...";
          setTimeout(()=>window.location.href = chrome.runtime.getURL("login.html"),1500);
        } else msgDiv.textContent=data.message || "Registration failed";
      } catch(e){ msgDiv.textContent="Error: "+e.message; }
    });

    switchLink.addEventListener("click", ()=>{
      window.location.href = chrome.runtime.getURL("login.html");
    });
  }
});
