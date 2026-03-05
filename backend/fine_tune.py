from transformers import AutoModelForCausalLM, AutoTokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

model_name = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare dataset (format: user prompt + bot response)
def prepare_data(file_path):
    dataset = load_dataset('text', data_files=file_path)
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=128)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset['train']

train_dataset = prepare_data('training_data.txt')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args (small for quick run; increase epochs for better results)
training_args = TrainingArguments(
    output_dir='./fine_tuned_model',
    num_train_epochs=3,  # More = better, but slower
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')