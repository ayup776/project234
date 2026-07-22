import os

# This project only uses PyTorch. Setting USE_TF=0 before any
# transformers/sentence-transformers import stops the library from even
# checking for a TensorFlow install, which avoids errors from a
# Keras 3 install being incompatible with transformers' TF integration.
os.environ["USE_TF"] = "0"

import hashlib
from flask import Flask, request, jsonify, render_template

from chatbot.bert_model import BertChatbot
from tts_module_dl import text_to_speech  # SpeechT5 (real neural TTS) -- was gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

app = Flask(__name__)

# Loaded once at startup -- BERT model + intents are kept in memory
# for the lifetime of the server so every request is fast.
chatbot = BertChatbot(os.path.join(BASE_DIR, "chatbot", "intents.json"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()

    if not user_text:
        return jsonify({"error": "Empty message"}), 400

    bot_response, intent, confidence = chatbot.get_response(user_text)

    # Deterministic filename based on content so repeated identical
    # responses reuse the same cached audio file instead of regenerating it.
    file_hash = hashlib.md5(bot_response.encode("utf-8")).hexdigest()[:12]
    audio_filename = f"response_{file_hash}.wav"
    audio_path = os.path.join(AUDIO_DIR, audio_filename)

    if not os.path.exists(audio_path):
        text_to_speech(bot_response, audio_path)

    return jsonify(
        {
            "response": bot_response,
            "intent": intent,
            "confidence": round(confidence, 3),
            "audio_url": f"/static/audio/{audio_filename}",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
