from trl import SFTConfig, SFTTrainer
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load Model
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    device_map="auto",
)

# 2. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# 3. Data Preparation
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
    "{% endif %}"
    "{% endfor %}"
)
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE


def formatting_prompts_func(example):
    # The tokenizer requires a list of dictionaries
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}


# Load JSONL file
dataset = load_dataset(
    "json", data_files="/content/finetune_data.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func)

train_val_split = dataset.train_test_split(
    test_size=0.1, seed=42)  # Added a seed for reproducibility
final_datasets = {
    'train': train_val_split['train'],
    'validation': train_val_split['test']  # Using 'validation' as the key
}
train_dataset = train_val_split["train"]
validation_dataset = train_val_split["validation"]

# 4. Configure and Run Training
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,      # Pass the formatted training set
    eval_dataset=validation_dataset,      # Pass the formatted validation set
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,  # Can make training 5x faster for short sequences.
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        # num_train_epochs = 1, # For longer training runs!
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB etc
        #  eval_strategy = "steps",
        #  eval_steps = 25,
        #  save_strategy = "steps",
        #  save_steps = 25,
    ),
)

# Start fine-tuning
trainer.train()

# Evaluation ------------------------------------------------------------------
validation_set = final_datasets['validation']

correct_predictions = 0
total_predictions = len(validation_set)

print("--- Running Evaluation ---")
for example in validation_set:
    # Get the prompt (system + user messages) and the correct answer
    # All messages except the last one
    prompt_messages = example['messages'][:-1]
    gold_answer = example['messages'][-1]['content']

    # Generate the prompt text
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt_text], return_tensors="pt").to("cuda")

    # Generate the model's response
    outputs = model.generate(**inputs, max_new_tokens=5,
                             pad_token_id=tokenizer.eos_token_id)

    # Decode and clean up the prediction
    # We slice the output to remove the prompt text we fed in
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # cleaned_prediction = prediction[len(prompt_text):].strip()
    cleaned_prediction = prediction[prediction.rfind('\n'):].strip()

    # Compare prediction to the gold answer
    if cleaned_prediction.lower() == gold_answer.lower():
        correct_predictions += 1

# Calculate and print the accuracy
accuracy = (correct_predictions / total_predictions) * 100
print("\n--- Evaluation Complete ---")
print(f"Correct Predictions: {correct_predictions}")
print(f"Total Predictions: {total_predictions}")
print(f"Accuracy: {accuracy:.2f}%")
