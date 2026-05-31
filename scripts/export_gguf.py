import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    model_name = "google/gemma-3-270m"
    tokenizer_path = "configs/tokenizer"
    adapter_path = "checkpoints/stage2/final"
    merged_output_dir = "checkpoints/merged"
    
    print(f"Loading extended tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    print(f"Loading base model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    base_model.resize_token_embeddings(len(tokenizer))
    
    if not os.path.exists(adapter_path):
        print(f"Error: Stage 2 adapter path {adapter_path} does not exist.")
        return
        
    print(f"Loading adapter weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {merged_output_dir}...")
    os.makedirs(merged_output_dir, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)
    
    print("\nModel successfully merged and saved!")
    print(f"You can now convert the merged model in '{merged_output_dir}' to GGUF format using llama.cpp:")
    print("python3 llama.cpp/convert_hf_to_gguf.py checkpoints/merged --outfile checkpoints/gemma-miditext-q8.gguf --outtype q8_0")

if __name__ == "__main__":
    main()
