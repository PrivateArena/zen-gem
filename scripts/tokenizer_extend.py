import os
import sys
from transformers import AutoTokenizer

def main():
    # Target model name
    model_name = "google/gemma-3-270m"
    
    print(f"Loading base tokenizer from: {model_name}")
    try:
        # Check if we need token or if we can load it directly
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Retrying with a fallback token or local config if available...")
        # Fallback to local execution or warning
        sys.exit(1)
        
    print(f"Original vocabulary size: {len(tokenizer)}")
    
    # 1. Define custom MIDIText tokens
    special_midi_tokens = [
        # Structural/Header
        "PPQ:480", "PPQ:960", "PPQ:120", "TS(4/4)", "TS(3/4)", "TS(6/8)", "TS(2/4)", "TS(12/8)",
        "<|score|>", "<|end|>",
        # Voices
        "V:1", "V:2", "V:3", "V:4", "V:5", "V:6", "V:7", "V:8",
        # Positioning / Operators
        "@@", "r", "R", " & ", " | ", "|",
        # Common bare duration tokens
        "/2", "/3", "/4", "/8", "/16", "3/2", "3/4", "2/3", "1/3"
    ]
    
    # Note names with accidentals (flat/sharp)
    notes = ["c", "d", "e", "f", "g", "a", "b"]
    for n in notes:
        special_midi_tokens.append(f"{n}#")
        special_midi_tokens.append(f"{n}_")
        special_midi_tokens.append(f"{n}-")
        special_midi_tokens.append(f"{n}+")
        special_midi_tokens.append(f"{n}'")
        special_midi_tokens.append(f"{n},")
        
    # Octaves
    for o in range(0, 10):
        special_midi_tokens.append(f"o{o}")
        
    # Common velocities (multiples of 5 + max 127)
    for v in range(0, 128, 5):
        special_midi_tokens.append(f"v{v}")
    special_midi_tokens.append("v127")
    
    # Relative velocity nudges
    for v in range(1, 20):
        special_midi_tokens.append(f"v+{v}")
        special_midi_tokens.append(f"v-{v}")
        
    # Stage 2 metadata conditional generation tags
    special_midi_tokens.extend([
        "<|tracks:1|>", "<|tracks:2|>", "<|tracks:3|>", "<|tracks:4|>",
        "<|ts:4/4|>", "<|ts:3/4|>", "<|ts:6/8|>",
    ])
    
    # Prefix tags
    prefix_tags = ["<|tempo:", "<|key:", "<|bars:"]
    special_midi_tokens.extend(prefix_tags)
    
    # Remove duplicates if any
    special_midi_tokens = sorted(list(set(special_midi_tokens)))
    
    # 2. Add tokens to the tokenizer
    num_added = tokenizer.add_tokens(special_midi_tokens)
    print(f"Added {num_added} tokens to the tokenizer.")
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token.")
        
    print(f"New vocabulary size: {len(tokenizer)}")
    
    # 3. Test token compression
    sample_text = "PPQ:480 t120.00 TS(4/4)\nV:1 o5 v100\n/2 [c# e g'] [f a c'] |\n"
    
    # Original tokenizer mock (reload from model_name)
    base_tok = AutoTokenizer.from_pretrained(model_name)
    orig_tokens = base_tok.tokenize(sample_text)
    new_tokens = tokenizer.tokenize(sample_text)
    
    print("\n--- Tokenization Comparison ---")
    print(f"Sample: {repr(sample_text)}")
    print(f"Original token count: {len(orig_tokens)}")
    print(f"Original tokens: {orig_tokens}")
    print(f"Extended token count: {len(new_tokens)}")
    print(f"Extended tokens: {new_tokens}")
    
    compression = (1.0 - (len(new_tokens) / len(orig_tokens))) * 100
    print(f"Compression ratio: {compression:.2f}% reduction in sequence length")
    
    # 4. Save extended tokenizer
    os.makedirs("configs/tokenizer", exist_ok=True)
    tokenizer.save_pretrained("configs/tokenizer")
    print("\nExtended tokenizer saved to configs/tokenizer")

if __name__ == "__main__":
    main()
