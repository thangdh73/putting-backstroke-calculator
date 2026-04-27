const form = document.querySelector("#calculator");
const resultCard = document.querySelector("#result");
const status = document.querySelector("#status");
const predicted = document.querySelector("#backstroke");
const elevation = document.querySelector("#elevation");
const backswing = document.querySelector("#backswing");
const downswing = document.querySelector("#downswing");
const repeat = document.querySelector("#repeat");
const targetRoll = document.querySelector("#target-roll");
const method = document.querySelector("#method");
const player = document.querySelector("#audio");
const download = document.querySelector("#download");
const submit = form.querySelector("button[type='submit']");

const setLoading = (loading) => {
  submit.disabled = loading;
  submit.textContent = loading ? "Generating..." : "Predict & Play";
};

const hideMessages = () => {
  resultCard.hidden = true;
  status.textContent = "";
  status.classList.remove("error");
};

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideMessages();
  setLoading(true);

  try {
    const payload = Object.fromEntries(new FormData(form).entries());
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    predicted.textContent = `${data.predicted_backstroke.toFixed(2)} ${data.unit_label}`;
    elevation.textContent = `${data.elevation_cm.toFixed(2)} cm`;
    targetRoll.textContent = `${data.target_roll_cm.toFixed(0)} cm`;
    backswing.textContent = `${data.swing.backswing_time.toFixed(3)} s`;
    downswing.textContent = `${data.swing.dsi_time.toFixed(3)} s`;
    repeat.textContent = `${data.repeat_count}x`;
    method.textContent = data.method;

    const src = `data:${data.audio.mime};base64,${data.audio.base64}`;
    player.src = src;
    download.href = src;
    download.download = data.audio.filename;
    resultCard.hidden = false;
  } catch (error) {
    status.textContent = error.message;
    status.classList.add("error");
  } finally {
    setLoading(false);
  }
});
