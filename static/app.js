const form = document.querySelector("#transcribeForm");
const statusBox = document.querySelector("#status");
const output = document.querySelector("#output");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  statusBox.textContent = "Transkripsiyon çalışıyor...";
  output.textContent = "";

  try {
    const response = await fetch("/api/transcribe", {
      method: "POST",
      body: new FormData(form),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    }
    statusBox.textContent = `Tamamlandı. Run ID: ${data.run_id}`;
    output.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    statusBox.textContent = "Hata";
    output.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
