import os
import re
import sys
import subprocess
import json

def check_syntax(miditext_str: str) -> tuple[bool, str]:
    """Invoke the Go validator program to verify syntax."""
    validator_path = "./miditext-validate"
    if not os.path.exists(validator_path):
        # Fallback to run main.go
        cmd = ["go", "run", "./cmd/miditext-validate/main.go"]
    else:
        cmd = [validator_path]
        
    try:
        result = subprocess.run(
            cmd,
            input=miditext_str,
            capture_output=True,
            text=True,
            timeout=5
        )
        return (result.returncode == 0, result.stderr.strip())
    except subprocess.TimeoutExpired:
        return (False, "Timeout during validation")
    except Exception as e:
        return (False, f"Exception during validation: {str(e)}")

def analyze_musicality(content: str) -> dict:
    """Analyze note density, pitch range, and repetition structure."""
    metrics = {
        "note_count": 0,
        "pitch_min": 128,
        "pitch_max": 0,
        "pitch_range": 0,
        "notes_per_channel": {},
        "unique_durations": set(),
        "bars": 0,
        "average_density": 0,
        "repetition_score": 0.0
    }
    
    # 1. Parse PPQ and bars
    ppq = 480
    ppq_match = re.search(r"PPQ:(\d+)", content)
    if ppq_match:
        ppq = int(ppq_match.group(1))
        
    beats_per_bar = 4
    ts_match = re.search(r"TS\((\d+)/\d+\)", content)
    if ts_match:
        beats_per_bar = int(ts_match.group(1))
        
    bar_ticks = ppq * beats_per_bar
    max_tick = 0
    ticks = re.findall(r"@(\d+)", content)
    if ticks:
        max_tick = max([int(t) for t in ticks])
    metrics["bars"] = (max_tick + bar_ticks - 1) // bar_ticks if bar_ticks > 0 else 0
    
    # 2. Extract note commands and pitches
    # Notes are lowercase letters optionally followed by sharps (#), flats (-/_), and octaves
    note_pattern = r"(?:v\d+)?(?:o\d+)?([a-g])([#\-_+\'^,]*)([\d/]*)"
    notes_found = re.findall(note_pattern, content)
    metrics["note_count"] = len(notes_found)
    
    # Pitches analysis (simple semitone offsets)
    note_bases = {'c': 0, 'd': 2, 'e': 4, 'f': 5, 'g': 7, 'a': 9, 'b': 11}
    current_octave = 5
    current_channel = 1
    
    # Track states per voice block
    # Split content by voice commands to assign notes to channels
    voice_blocks = re.split(r"(V:\d+)", content)
    
    for i in range(1, len(voice_blocks), 2):
        v_tag = voice_blocks[i]
        v_num = int(re.search(r"\d+", v_tag).group())
        block_content = voice_blocks[i+1]
        
        metrics["notes_per_channel"][v_num] = 0
        
        # Track active octaves and accidentals in block
        octaves = re.findall(r"o(\d+)", block_content)
        block_notes = re.findall(note_pattern, block_content)
        
        metrics["notes_per_channel"][v_num] = len(block_notes)
        
        # Track pitches
        for base, modifier, dur in block_notes:
            pitch_val = note_bases[base]
            # Simple octave shift check
            oct_match = re.search(r"o(\d+)", block_content)
            octave = int(oct_match.group(1)) if oct_match else 5
            
            # Modifier adjustments
            alter = 0
            alter += modifier.count('#')
            alter += modifier.count('+')
            alter -= modifier.count('-')
            alter -= modifier.count('_')
            
            pitch = 12 * octave + pitch_val + alter
            
            if pitch < metrics["pitch_min"]:
                metrics["pitch_min"] = pitch
            if pitch > metrics["pitch_max"]:
                metrics["pitch_max"] = pitch
                
            if dur:
                metrics["unique_durations"].add(dur)
                
    if metrics["pitch_max"] >= metrics["pitch_min"]:
        metrics["pitch_range"] = metrics["pitch_max"] - metrics["pitch_min"]
        
    metrics["unique_durations"] = list(metrics["unique_durations"])
    
    # Average note density (notes per bar)
    if metrics["bars"] > 0:
        metrics["average_density"] = metrics["note_count"] / metrics["bars"]
        
    # Repetition score (based on bar repetition)
    bars_list = content.split(" | ")
    if len(bars_list) > 1:
        unique_bars = len(set(bars_list))
        metrics["repetition_score"] = 1.0 - (unique_bars / len(bars_list))
        
    return metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 eval_syntax.py <generated_output_jsonl_or_text_file>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: file {input_file} not found")
        sys.exit(1)
        
    print(f"Evaluating samples in {input_file}...")
    
    samples = []
    if input_file.endswith(".jsonl"):
        with open(input_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "generated" in data:
                            samples.append(data["generated"])
                        elif "text" in data:
                            samples.append(data["text"])
                    except Exception:
                        pass
    else:
        # Read as single plain text file containing a generated MIDIText score
        with open(input_file, "r") as f:
            samples.append(f.read())
            
    if not samples:
        print("No samples found to evaluate.")
        sys.exit(1)
        
    print(f"Found {len(samples)} samples. Starting evaluation...")
    
    valid_count = 0
    total_notes = 0
    total_bars = 0
    pitch_ranges = []
    repetition_scores = []
    
    for idx, s in enumerate(samples):
        # Extract score content (in case tags are present)
        score_content = s
        score_start = s.find("<|score|>")
        if score_start != -1:
            score_end = s.find("<|end|>", score_start)
            if score_end != -1:
                score_content = s[score_start + len("<|score|>"):score_end].strip()
            else:
                score_content = s[score_start + len("<|score|>"):].strip()
                
        is_valid, err = check_syntax(score_content)
        if is_valid:
            valid_count += 1
            m = analyze_musicality(score_content)
            total_notes += m["note_count"]
            total_bars += m["bars"]
            pitch_ranges.append(m["pitch_range"])
            repetition_scores.append(m["repetition_score"])
        else:
            print(f"Sample {idx} Invalid! Error: {err}")
            
    validity_rate = (valid_count / len(samples)) * 100
    avg_notes = total_notes / valid_count if valid_count > 0 else 0
    avg_bars = total_bars / valid_count if valid_count > 0 else 0
    avg_pitch_range = sum(pitch_ranges) / len(pitch_ranges) if pitch_ranges else 0
    avg_repetition = sum(repetition_scores) / len(repetition_scores) if repetition_scores else 0
    
    print("\n================ Evaluation Summary ================")
    print(f"Syntax Validity Rate: {validity_rate:.2f}% ({valid_count}/{len(samples)})")
    print(f"Average Note Count:   {avg_notes:.1f}")
    print(f"Average Bar Count:    {avg_bars:.1f}")
    print(f"Average Pitch Spread: {avg_pitch_range:.1f} semitones")
    print(f"Average Repetition:   {avg_repetition:.2%}")
    print("====================================================\n")

if __name__ == "__main__":
    main()
