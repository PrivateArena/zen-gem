import os
import re
import json
import random

def estimate_tempo_and_bars(content):
    # Estimate PPQ
    ppq = 480
    ppq_match = re.search(r"PPQ:(\d+)", content)
    if ppq_match:
        ppq = int(ppq_match.group(1))
        
    # Estimate tempo (BPM)
    tempo = 120
    tempo_match = re.search(r"t(\d+(?:\.\d+)?)", content)
    if tempo_match:
        tempo = int(float(tempo_match.group(1)))
        
    # Estimate tracks/voices count
    tracks = len(re.findall(r"V:\d+", content))
    if tracks == 0:
        tracks = 1
        
    # Estimate total bars based on max tick @N
    max_tick = 0
    ticks = re.findall(r"@(\d+)", content)
    if ticks:
        max_tick = max([int(t) for t in ticks])
        
    # Beats per bar is 4 by default
    beats_per_bar = 4
    ts_match = re.search(r"TS\((\d+)/\d+\)", content)
    if ts_match:
        beats_per_bar = int(ts_match.group(1))
        
    bar_ticks = ppq * beats_per_bar
    if bar_ticks == 0:
        bar_ticks = 480 * 4
    bars = (max_tick + bar_ticks - 1) // bar_ticks
    if bars == 0:
        bars = 4
        
    return tempo, bars, tracks

def main():
    random.seed(42)
    
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        print(f"Error: raw data directory {raw_dir} does not exist")
        return
        
    print("Scanning raw files...")
    all_files = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith(".txt"):
                all_files.append(os.path.join(root, f))
                
    print(f"Found {len(all_files)} raw MIDIText files.")
    
    # Shuffle files to mix sources
    random.shuffle(all_files)
    
    samples = []
    skipped_too_short = 0
    skipped_too_long = 0
    
    # Quick character-level filtering (roughly 64 to 3500 tokens)
    min_chars = 200
    max_chars = 15000
    
    print("Processing and formatting files...")
    for idx, filepath in enumerate(all_files):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
        if not content:
            continue
            
        char_len = len(content)
        if char_len < min_chars:
            skipped_too_short += 1
            continue
        if char_len > max_chars:
            skipped_too_long += 1
            continue
            
        tempo, bars, tracks = estimate_tempo_and_bars(content)
        
        # Format Stage 1 (CLM)
        clm_text = f"<|score|>\n{content}\n<|end|>"
        
        # Format Stage 2 (Conditional Fine-Tuning)
        cond_text = f"<|tracks:{tracks}|><|tempo:{tempo}|><|bars:{bars}|><|score|>\n{content}\n<|end|>"
        
        samples.append({
            "stage1": clm_text,
            "stage2": cond_text
        })
        
    print(f"\nProcessing finished.")
    print(f"Total valid samples: {len(samples)}")
    print(f"Skipped too short (<{min_chars} chars): {skipped_too_short}")
    print(f"Skipped too long (>{max_chars} chars): {skipped_too_long}")
    
    if len(samples) == 0:
        print("No valid samples found. Exiting.")
        return
        
    # Split into train/val (98% / 2%)
    val_count = max(1, int(len(samples) * 0.02))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    
    # Save datasets
    os.makedirs("data/final", exist_ok=True)
    
    # Save Stage 1 datasets
    with open("data/final/train_stage1.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps({"text": s["stage1"]}) + "\n")
            
    with open("data/final/val_stage1.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps({"text": s["stage1"]}) + "\n")
            
    # Save Stage 2 datasets
    with open("data/final/train_stage2.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps({"text": s["stage2"]}) + "\n")
            
    with open("data/final/val_stage2.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps({"text": s["stage2"]}) + "\n")
            
    print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
