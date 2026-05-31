import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import load_dataset
from peft import PeftModel

def main():
    # Paths
    model_name = "google/gemma-3-270m"
    tokenizer_path = "configs/tokenizer"
    stage1_adapter_path = "checkpoints/stage1/final"
    train_file = "data/final/train_stage2.jsonl"
    val_file = "data/final/val_stage2.jsonl"
    output_dir = "checkpoints/stage2"
    
    print(f"Loading extended tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Loading base model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True
    )
    # Resize embeddings for our custom tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load Stage 1 adapter weights
    print(f"Loading Stage 1 adapter from {stage1_adapter_path}...")
    if os.path.exists(stage1_adapter_path):
        model = PeftModel.from_pretrained(base_model, stage1_adapter_path, is_trainable=True)
        print("Successfully loaded Stage 1 adapter weights for continued training.")
    else:
        print(f"Warning: Stage 1 adapter not found at {stage1_adapter_path}. Initializing empty PEFT configuration...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
        
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
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=5e-5,               # Lower learning rate for fine-tuning stage
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        use_cpu=True,
        bf16=False,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        save_total_limit=2,
        report_to="none"
    )
    
    # Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=data_collator,
        args=training_args,
    )
    
    # Run training
    print("Starting training...")
    trainer.train()
    
    print(f"Saving fine-tuned Stage 2 model to {output_dir}/final...")
    trainer.save_model(os.path.join(output_dir, "final"))
    print("Stage 2 training completed!")

if __name__ == "__main__":
    main()
