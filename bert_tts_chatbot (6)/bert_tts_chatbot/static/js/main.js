const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const chatWindow = document.getElementById("chat-window");
const emptyState = document.getElementById("empty-state");
const statusDot = document.getElementById("status-dot");
const statusLabel = document.getElementById("status-label");
const resetBtn = document.getElementById("reset-btn");
const intentsList = document.getElementById("intents-list");
const intentCount = document.getElementById("intent-count");

// Known intents shown in the sidebar. Kept in sync manually with
// chatbot/intents.json -- update this list if you add/remove intents there.
const KNOWN_INTENTS = [
  "greeting", "wellbeing", "goodbye", "thanks", "identity",
  "weather", "help", "goodnight", "joke", "time", "age",
  "compliment", "mood_check",
];

function renderIntentsList() {
  intentsList.innerHTML = "";
  KNOWN_INTENTS.forEach((tag) => {
    const li = document.createElement("li");
    li.textContent = tag;
    li.dataset.tag = tag;
    intentsList.appendChild(li);
  });
  intentCount.textContent = `(${KNOWN_INTENTS.length})`;
}

function highlightIntent(tag) {
  document.querySelectorAll(".intents-list li").forEach((li) => {
    li.style.background = li.dataset.tag === tag ? "" : "";
  });
  const match = intentsList.querySelector(`li[data-tag="${tag}"]`);
  if (match) {
    match.classList.add("active-flash");
    match.style.background = "var(--accent-soft)";
    match.style.color = "var(--accent-strong)";
    setTimeout(() => {
      match.style.background = "";
      match.style.color = "";
    }, 1400);
  }
}

function confidenceTier(score) {
  if (score >= 0.8) return "high";
  if (score >= 0.6) return "mid";
  return "low";
}

function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function clearEmptyState() {
  if (emptyState && emptyState.parentNode) {
    emptyState.remove();
  }
}

function appendUserMessage(text) {
  clearEmptyState();
  const row = document.createElement("div");
  row.className = "msg-row user";
  row.innerHTML = `<div class="msg-bubble"></div>`;
  row.querySelector(".msg-bubble").textContent = text;
  chatWindow.appendChild(row);
  scrollToBottom();
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row bot typing-row";
  row.id = "typing-indicator";
  row.innerHTML = `
    <div class="msg-bubble">
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
      <span class="typing-dot"></span>
    </div>`;
  chatWindow.appendChild(row);
  scrollToBottom();
}

function removeTypingIndicator() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function appendBotMessage({ response, intent, confidence, audio_url }) {
  const row = document.createElement("div");
  row.className = "msg-row bot";

  const tier = confidenceTier(confidence);
  const pct = Math.round(confidence * 100);

  row.innerHTML = `
    <div class="msg-bubble"></div>
    <div class="msg-meta">
      <span class="intent-tag">${intent}</span>
      <span class="confidence-pill ${tier}">${pct}%</span>
    </div>
    <div class="audio-player" hidden>
      <button type="button" class="play-btn" aria-label="Play response audio">▶</button>
      <div class="waveform">
        <span></span><span></span><span></span><span></span><span></span>
      </div>
    </div>
  `;
  row.querySelector(".msg-bubble").textContent = response;
  chatWindow.appendChild(row);

  if (audio_url) {
    const player = row.querySelector(".audio-player");
    const playBtn = row.querySelector(".play-btn");
    const waveform = row.querySelector(".waveform");
    const audio = new Audio(audio_url);
    player.hidden = false;

    playBtn.addEventListener("click", () => {
      if (audio.paused) {
        audio.play();
      } else {
        audio.pause();
      }
    });

    audio.addEventListener("play", () => {
      playBtn.textContent = "❚❚";
      waveform.classList.add("playing");
    });
    const reset = () => {
      playBtn.textContent = "▶";
      waveform.classList.remove("playing");
    };
    audio.addEventListener("pause", reset);
    audio.addEventListener("ended", reset);

    // Autoplay the reply; browsers may block this until the user has
    // interacted with the page once, which they have by sending a message.
    audio.play().catch(() => {
      /* Autoplay blocked -- user can press the play button manually. */
    });
  }

  highlightIntent(intent);
  scrollToBottom();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  appendUserMessage(message);
  input.value = "";
  input.disabled = true;
  sendBtn.disabled = true;
  statusDot.classList.add("busy");
  statusLabel.textContent = "Thinking…";
  appendTypingIndicator();

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();

    removeTypingIndicator();

    if (data.error) {
      appendBotMessage({
        response: `Error: ${data.error}`,
        intent: "error",
        confidence: 0,
        audio_url: null,
      });
      return;
    }

    appendBotMessage(data);
  } catch (err) {
    removeTypingIndicator();
    appendBotMessage({
      response: "Couldn't reach the server. Is app.py still running?",
      intent: "error",
      confidence: 0,
      audio_url: null,
    });
    console.error(err);
  } finally {
    input.disabled = false;
    sendBtn.disabled = false;
    statusDot.classList.remove("busy");
    statusLabel.textContent = "Ready";
    input.focus();
  }
});

resetBtn.addEventListener("click", () => {
  chatWindow.innerHTML = "";
  const el = document.createElement("div");
  el.className = "empty-state";
  el.id = "empty-state";
  el.innerHTML = `
    <span class="empty-mark" aria-hidden="true"></span>
    <p>Send a message to start. The assistant replies in text and speech.</p>
  `;
  chatWindow.appendChild(el);
});

renderIntentsList();
input.focus();
