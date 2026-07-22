"""
Deep-learning text-to-speech using SpeechT5 (Microsoft), the same family
of architecture as FastSpeech2: a transformer encoder turns text into a
hidden representation, a transformer decoder (+ post-net) predicts a
mel-spectrogram, and a neural vocoder (HiFi-GAN) turns that spectrogram
into a waveform. Unlike gTTS, this is a real neural network running
locally -- no external API call.

Pipeline:
    text -> SpeechT5Processor (tokenize)
         -> SpeechT5ForTextToSpeech (transformer encoder-decoder -> mel-spectrogram)
         -> SpeechT5HifiGan (vocoder: mel-spectrogram -> waveform)

Usage is a drop-in replacement for tts_module.text_to_speech():

    from tts_module_dl import text_to_speech
    text_to_speech("Hello there", "static/audio/out.wav")

First run downloads the models (~600MB total) and a speaker embedding
dataset once; both are cached afterwards.
"""

import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

_MODEL_NAME = "microsoft/speecht5_tts"
_VOCODER_NAME = "microsoft/speecht5_hifigan"
_SPEAKER_EMBEDDING_CACHE = os.path.join(os.path.dirname(__file__), "speaker_embedding.pt")

_device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[tts_module_dl] Loading SpeechT5 on {_device} ...")
_processor = SpeechT5Processor.from_pretrained(_MODEL_NAME)
_model = SpeechT5ForTextToSpeech.from_pretrained(_MODEL_NAME).to(_device)
_vocoder = SpeechT5HifiGan.from_pretrained(_VOCODER_NAME).to(_device)
_model.eval()
_vocoder.eval()


def _load_speaker_embedding() -> torch.Tensor:
    """
    SpeechT5 is a multi-speaker model, so it needs a speaker embedding
    (a vector describing voice characteristics) alongside the text.
    We pull one example voice from the CMU ARCTIC x-vector dataset the
    first time, then cache it locally so we don't re-download later.
    """
    if os.path.exists(_SPEAKER_EMBEDDING_CACHE):
        return torch.load(_SPEAKER_EMBEDDING_CACHE)

    from datasets import load_dataset

    print("[tts_module_dl] Fetching a speaker embedding (first run only) ...")
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    # Index 7306 is a commonly used clear female voice; pick any index
    # from 0..len(embeddings_dataset) to change the voice.
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    torch.save(speaker_embedding, _SPEAKER_EMBEDDING_CACHE)
    return speaker_embedding


_speaker_embedding = _load_speaker_embedding().to(_device)


@torch.no_grad()
def text_to_speech(text: str, output_path: str, lang: str = "en") -> str:
    """
    Runs real transformer-based TTS inference and writes a 16kHz mono
    WAV file to output_path. `lang` is accepted for interface
    compatibility with tts_module.py but SpeechT5 here is English-only.
    """
    inputs = _processor(text=text, return_tensors="pt").to(_device)

    speech = _model.generate_speech(
        inputs["input_ids"], _speaker_embedding, vocoder=_vocoder
    )

    sf.write(output_path, speech.cpu().numpy(), samplerate=16000)
    return output_path


if __name__ == "__main__":
    # Quick manual test: python tts_module_dl.py
    out = text_to_speech("Hello, this is a deep learning text to speech test.", "test_output.wav")
    print(f"Wrote {out}")
