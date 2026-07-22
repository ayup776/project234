"""
BERT-based chatbot core -- now using sentence-transformers instead of
raw BertModel for the embedding step.

Why the change: plain BertModel (mean-pooled last hidden state, no
fine-tuning) suffers from a well-known "anisotropy" problem -- its
sentence embeddings cluster tightly together in vector space, so
cosine similarity between *unrelated* sentences stays artificially
high (often 0.5-0.7+). That made the original version confidently
match things like "say numbers from 1 to 10" to the "weather" intent
at 0.596 similarity, above the 0.55 threshold, even though the two
have nothing to do with each other.

sentence-transformers models (like all-MiniLM-L6-v2 used here) are
still BERT-family transformers, but fine-tuned via contrastive
learning specifically so that cosine similarity reflects actual
semantic similarity. Same architecture family, much more reliable
matching, and it's smaller/faster than bert-base-uncased too.

Everything else about the design is unchanged:
 1. Every training "pattern" in intents.json is embedded once at startup.
 2. An incoming user message is embedded the same way.
 3. We pick the pattern with the highest cosine similarity and return
    a response from that pattern's intent, or a fallback if nothing
    is a confident enough match.
"""

import json
import random
from sentence_transformers import SentenceTransformer, util


class BertChatbot:
    def __init__(self, intents_path: str, model_name: str = "all-MiniLM-L6-v2",
                 confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold

        print(f"[BertChatbot] Loading sentence-transformer '{model_name}' ...")
        self.model = SentenceTransformer(model_name)

        with open(intents_path, "r", encoding="utf-8") as f:
            self.intents = json.load(f)["intents"]

        self.patterns = []
        self.pattern_to_intent = []
        for intent in self.intents:
            for pattern in intent["patterns"]:
                self.patterns.append(pattern)
                self.pattern_to_intent.append(intent["tag"])

        print(f"[BertChatbot] Embedding {len(self.patterns)} training patterns ...")
        self.pattern_embeddings = self.model.encode(self.patterns, convert_to_tensor=True)
        print("[BertChatbot] Ready.")

    def get_response(self, user_text: str):
        """Returns (response_text, matched_intent_tag, confidence_score)."""
        user_embedding = self.model.encode(user_text, convert_to_tensor=True)
        similarities = util.cos_sim(user_embedding, self.pattern_embeddings)[0]

        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        matched_tag = self.pattern_to_intent[best_idx]

        if best_score < self.confidence_threshold:
            return (
                "Sorry, I didn't quite understand that. Could you rephrase it?",
                "unknown",
                best_score,
            )

        for intent in self.intents:
            if intent["tag"] == matched_tag:
                return random.choice(intent["responses"]), matched_tag, best_score

        return "I'm not sure how to respond to that.", "unknown", best_score
