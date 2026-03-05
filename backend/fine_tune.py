"""
finetune.py  —  Fine-tune DialoGPT-medium on stress-support conversations.

Run once before starting the server:
    python finetune.py

Output: ./fine_tuned_model/   (used automatically by chatmodel.py)
"""

import os, json, torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "microsoft/DialoGPT-medium"   # medium gives better quality than small
OUTPUT_DIR   = "./fine_tuned_model"
DATA_FILE    = "./training_data/training_data.txt"
MAX_LENGTH   = 256
EPOCHS       = 5          # increase to 10 for better results (takes longer)
BATCH_SIZE   = 2          # increase if you have a GPU with >4 GB VRAM
LR           = 5e-5
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading base model: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
model.config.pad_token_id = tokenizer.eos_token_id

# ── Load & tokenise data ──────────────────────────────────────────────────────
def load_raw(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines

raw_texts = load_raw(DATA_FILE)
print(f"Loaded {len(raw_texts)} training lines.")

def tokenise(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )

ds = Dataset.from_dict({"text": raw_texts})
ds = ds.map(tokenise, batched=True, remove_columns=["text"])
ds = ds.with_format("torch")

# ── Training ──────────────────────────────────────────────────────────────────
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,   # effective batch = BATCH_SIZE * 4
    learning_rate=LR,
    warmup_steps=20,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    fp16=torch.cuda.is_available(),   # use mixed precision on GPU
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collator,
)

print("Starting fine-tuning…")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done! Run server.py to start the chatbot.")