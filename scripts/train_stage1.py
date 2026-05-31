import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

def main():
    # Paths
    model_name = "google/gemma-3-270m"
    tokenizer_path = "configs/tokenizer"
    train_file = "data/final/train_stage1.jsonl"
    val_file = "data/final/val_stage1.jsonl"
    output_dir = "checkpoints/stage1"
    
    print(f"Loading extended tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Loading base model {model_name}...")
    # Load in float32 for CPU compatibility, or bfloat16 if supported and requested
    # Ryzen AI MAX CPU supports bfloat16/avx512 natively.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Resize embeddings for our custom tokens
    print(f"Resizing model embeddings to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer))
    
    # LoRA Configuration
    print("Setting up PEFT LoRA adapter...")
    lora_config = LoraConfig(
        r=32,                  # Moderate rank to balance CPU performance and learning capacity
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print(f"Loading datasets: {train_file}, {val_file}...")
    dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": val_file
    })
    
    # Tokenization preprocessing function
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=2048)
        
    print("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )
    
    # Collator for causal language modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training Arguments optimized for local Ryzen CPU training
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,      # Small batch size to minimize CPU memory footprint per step
        gradient_accumulation_steps=8,     # Accumulate steps to simulate batch size of 16
        num_train_epochs=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        use_cpu=True,                      # Force CPU execution
        bf16=False,                        # Keep False on standard CPU PyTorch build unless AMP is configured
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        optim="adamw_torch",               # Standard torch AdamW (reliable on CPU)
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_total_limit=2,
        report_to="none"                   # Disable third-party logging platforms
    )
    
    # Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        dataset_text_field="text",         # Note: tokenized_dataset will bypass raw text mapping
        max_seq_length=2048,
        data_collator=data_collator,
        args=training_args,
    )
    
    # Run training
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned adapter to {output_dir}/final...")
    trainer.save_model(os.path.join(output_dir, "final"))
    print("Stage 1 training completed!")

if __name__ == "__main__":
    main()
