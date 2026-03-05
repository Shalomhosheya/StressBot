"""
chatmodel.py  —  100% local inference. No external API needed.
Uses fine-tuned DialoGPT-medium (falls back to base model if not yet fine-tuned).
"""

import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Model path ────────────────────────────────────────────────────────────────
_FINE_TUNED = os.path.join(os.path.dirname(__file__), "fine_tuned_model")
_BASE       = "microsoft/DialoGPT-medium"
MODEL_PATH  = _FINE_TUNED if os.path.isdir(_FINE_TUNED) else _BASE

print(f"[Serenity] Loading model: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
model.eval()
model.config.pad_token_id = tokenizer.eos_token_id
print("[Serenity] Model ready.")

# ── Intent classifier ─────────────────────────────────────────────────────────
INTENT_RE = {
    "crisis":     re.compile(r"\b(suicid|kill myself|end my life|self.harm|hurt myself|want to die|no reason to live|not worth living)\b", re.I),
    "breathing":  re.compile(r"\b(breath|panic|can'?t breathe|hyperventilat|anxious right now|calm down)\b", re.I),
    "cbt":        re.compile(r"\b(overthink|spiral|catastroph|worst.case|negative thought|stuck in my head|keep thinking)\b", re.I),
    "validation": re.compile(r"\b(nobody understands|feel alone|so stress|overwhelm|exhausted|burnt.?out|no one cares)\b", re.I),
}

CRISIS_REPLY = (
    "I hear you, and I'm taking this seriously. "
    "Please contact a crisis helpline right now — free, confidential, 24/7.\n\n"
    "US: call or text 988\n"
    "International: https://www.befrienders.org\n\n"
    "You don't have to face this alone. I'm still here."
)

BREATHING_REPLY = (
    "Let's slow this down together. Try box breathing:\n"
    "Breathe IN for 4 counts\n"
    "HOLD for 4 counts\n"
    "Breathe OUT for 4 counts\n"
    "HOLD for 4 counts\n\n"
    "Repeat 4-6 times. Your nervous system will start to calm. "
    "How are you feeling after a few rounds?"
)


def classify_intent(text: str) -> str:
    for intent, pattern in INTENT_RE.items():
        if pattern.search(text):
            return intent
    return "general"


def generate_response(user_input, chat_history_ids=None):
    intent = classify_intent(user_input)

    if intent == "crisis":
        return CRISIS_REPLY, chat_history_ids

    if intent == "breathing" and chat_history_ids is None:
        return BREATHING_REPLY, chat_history_ids

    new_ids = tokenizer.encode(
        user_input + tokenizer.eos_token,
        return_tensors="pt",
    )

    bot_input_ids = (
        torch.cat([chat_history_ids, new_ids], dim=-1)
        if chat_history_ids is not None
        else new_ids
    )

    if bot_input_ids.shape[-1] > 900:
        bot_input_ids = bot_input_ids[:, -900:]

    with torch.inference_mode():
        output_ids = model.generate(
            bot_input_ids,
            max_new_tokens=80,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.90,
            top_k=50,
            temperature=0.75,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    response_ids = output_ids[:, bot_input_ids.shape[-1]:][0]
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

    if not response:
        response = "I'm here and listening. Can you tell me more about how you're feeling?"

    return response, output_ids