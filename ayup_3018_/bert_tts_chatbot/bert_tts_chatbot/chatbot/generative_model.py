"""
Open-domain fallback for the chatbot.

BertChatbot (bert_model.py) only knows how to respond to the intents
defined in intents.json -- anything outside those topics used to just
get "Sorry, I didn't quite understand that." This module adds a real
generative language model as a fallback, so the assistant can respond
to arbitrary questions instead of only a fixed set of canned replies.

Model: Qwen2.5-0.5B-Instruct -- chosen specifically because it's small
enough to run inference on CPU in a few seconds (unlike multi-billion
parameter chat models, which would take much longer per reply without
a GPU), while still being instruction-tuned for conversational use.

This is a genuinely different kind of model than BertChatbot: BERT
here is an *encoder* used for matching meaning, not for generating
text. This is a *decoder* (causal language model) that produces new
tokens one at a time based on everything generated so far.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep answers short and natural "
    "to speak aloud -- 1 to 3 sentences unless the user asks for more detail."
)


class GenerativeChatbot:
    def __init__(self, model_name: str = _MODEL_NAME, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[GenerativeChatbot] Loading '{model_name}' on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[GenerativeChatbot] Ready.")

    @torch.no_grad()
    def generate_reply(self, user_text: str, max_new_tokens: int = 100) -> str:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
        reply = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return reply or "I'm not sure how to answer that."
