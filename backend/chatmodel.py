# chatmodel.py (or whatever you renamed it to)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/DialoGPT-small"          # or your fine-tuned path
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# We'll also set pad_token = eos_token to avoid the warning
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def generate_response(user_input, chat_history_ids=None):
    # chat_history_ids should be a 2D tensor or None
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    output_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        # recommended settings for DialoGPT
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.75,
        no_repeat_ngram_size=3,   # helps prevent repetition
    )

    # Extract only the newly generated part
    response_ids = output_ids[:, bot_input_ids.shape[-1]:][0]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)

    # Return updated history as 2D tensor (keeps dimension consistent)
    updated_history_ids = output_ids  # already 2D: [1, full_length]

    return response, updated_history_ids