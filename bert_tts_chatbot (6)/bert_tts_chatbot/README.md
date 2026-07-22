# BERT Voice Chatbot (Flask)

A text-in, voice-out chatbot: user messages are understood using BERT
sentence embeddings, matched against an intents dataset, and the
chosen response is spoken back using a real neural text-to-speech
model (SpeechT5), all served through a small Flask web app.

## How it works

1. **Understanding (BERT):** `chatbot/bert_model.py` loads
   `bert-base-uncased`, embeds every example phrase in
   `chatbot/intents.json` once at startup, and embeds each incoming
   user message the same way. The intent whose example phrase is
   closest (cosine similarity) to the user's message wins. This needs
   no training step, since it's just using BERT as a semantic
   encoder.
2. **Response selection:** once an intent is matched (above a
   confidence threshold), a random response from that intent's
   response list is chosen.
3. **Speech (SpeechT5):** `tts_module_dl.py` runs the chosen response
   text through SpeechT5 (transformer encoder-decoder -> mel-spectrogram)
   and a HiFi-GAN vocoder (mel-spectrogram -> waveform), saving a real
   `.wav` file under `static/audio/`. This is an actual neural network
   doing inference locally, not an external API call.
4. **Flask glue:** `app.py` exposes `POST /chat`, which takes
   `{"message": "..."}` and returns:
   ```json
   {
     "response": "Hello! How can I help you today?",
     "intent": "greeting",
     "confidence": 0.812,
     "audio_url": "/static/audio/response_ab12cd34ef56.wav"
   }
   ```
5. **Frontend:** `templates/index.html` + `static/js/main.js` give a
   simple chat window that plays the returned audio automatically.

## Project structure

```
bert_tts_chatbot/
├── app.py                  # Flask server / routes
├── tts_module_dl.py         # SpeechT5 + HiFi-GAN wrapper (real neural TTS)
├── chatbot/
│   ├── bert_model.py       # BERT embedding + matching logic
│   └── intents.json        # training patterns + responses (edit this!)
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── audio/              # generated speech files land here
├── speaker_embedding.pt    # cached voice profile (created on first run)
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first run downloads `bert-base-uncased` (~440MB), `speecht5_tts`
and `speecht5_hifigan` (~600MB combined), plus one speaker embedding
from a small voice-embedding dataset. Needs internet access once;
everything is cached locally afterward. Budget a few GB of disk space.

## Run

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser, type a message,
and the bot will reply in the chat window and speak the response out
loud. On CPU, expect each new (uncached) response to take a few
seconds to synthesize; repeated identical responses reuse the cached
audio file instantly.

## Customizing the chatbot

Everything the bot can talk about lives in `chatbot/intents.json`.
To teach it a new topic, add a new object:

```json
{
  "tag": "hours",
  "patterns": ["What are your hours", "When are you open"],
  "responses": ["We're open 9am to 6pm, Monday through Friday."]
}
```

More/varied `patterns` per intent = better matching, since BERT
compares the user's message against every example phrase.

## Changing the bot's voice

`tts_module_dl.py` picks one fixed voice from the CMU ARCTIC
x-vector dataset (index `7306`). To try a different voice, change
that index, delete `speaker_embedding.pt` so it re-fetches, and
restart the server.

## Things worth upgrading later

- **Better matching:** swap the mean-pooled BERT embeddings for
  `sentence-transformers` (e.g. `all-MiniLM-L6-v2`), which is
  purpose-built for semantic similarity and much faster.
- **Real classification:** fine-tune `BertForSequenceClassification`
  on your intents instead of nearest-neighbor matching, if you have
  enough labeled examples per intent.
- **Conversation memory:** the current bot is stateless per message;
  add a session-based chat history if you want context across turns.
- **GPU:** both BERT and SpeechT5 auto-detect and use `cuda` if
  available, for much faster inference -- no code changes needed.
