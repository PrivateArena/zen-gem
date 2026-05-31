To transform the lightweight **Gemma 3 270M** into a production-grade music composition engine, we need to address its fundamental constraint: it has a limited capacity for abstract reasoning but an incredible aptitude for dense, ultra-fast structural mapping.

Because you have an endless supply of perfect, lossless 2-way MIDI-to-MIDIText data, your task is fundamentally a **Domain-Specific Continual Pre-training** project rather than a simple instruction-following fine-tune.

MIDIText Source: file:///media/jang/home/Deve/zenmidi/pkg/miditext/
MIDIText Notation: file:///media/jang/home/Deve/zenmidi/pkg/miditext/MIDITEXT_NOTATION_SKILL.md
Professional MIDI datasets:
file:///media/jang/home/Deve/midi/ninsheetmusic/ - Simple but very high quality, good for fast generation. It is also small enough to be processed.
file:///media/jang/home/Deve/midi/vgmusic/ - Video games, old, but very good in terms of structure and harmony.
file:///media/jang/home/Deve/midi/vgmusicnew/ - New video games, very high quality.
file:///media/jang/home/Deve/midi/khinsider - Huge collection of video game music.
file:///media/jang/home/Deve/midi/downloaded/ - MIDI files downloaded from various sources.
file:///media/jang/home/Deve/midi/transcribed/ - Extremely high quality performance from Youtuber and professional, totally humanized performance.
file:///media/jang/home/Deve/midi/netcavy/
file:///media/jang/home/Deve/midi/musescore - Giant database of millions of professional grade MIDI scores.
file:///media/jang/home/Deve/midi/midishow - Chinese made MIDI scores.

With those, we need to import /media/jang/home/Deve/zenmidi/pkg/miditext/ to convert them to MIDIText to create our training dataset.

---

## Phase 1: Data Architecture & Tokenizer Optimization

The default Gemma 3 tokenizer is optimized for multilingual text and code. Left unchanged, it will split a dense token string like `[c# e_ g+2]/4` into 6 or 7 distinct sub-word tokens, massively inflating your context windows and inviting syntactic hallucinations.

### 1. Register Custom Tokens

Before running any training loops, you must modify the tokenizer. You must force the vocabulary to treat critical MIDIText control syntax as single, atomic tokens.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", token="YOUR_HF_TOKEN")

# Add structural tokens that must never be split into sub-words
special_midi_tokens = [
    "PPQ:480", "TS(4/4)", "TS(3/4)", "TS(6/8)", 
    "V:1", "V:2", "V:3", "V:4", "V:5", "V:6", "V:7", "V:8",
    "|", "@@", "r", "R", "/2", "/4", "/8", "/3", "3/2"
]

tokenizer.add_tokens(special_midi_tokens)
# Ensure pad token is properly handled
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

```

### 2. Chunking the Data by Musical Phrasing

Do not slice your billions of musical scores by arbitrary token counts. Instead, use your converter to segment the data into exact **4-bar, 8-bar, or 16-bar logical loops**.

Wrap your dataset into a structured, unified "Context-to-Target" template so the model explicitly connects music theory parameters to the actual output notes:

```json
{
  "instruction": "Generate a soulful Rhodes chord progression over 4 bars.",
  "input": "<style>Neo-Soul</style><key>Dmin7</key><bpm>88</bpm><bars>4</bars>",
  "output": "PPQ:480 t88.00 TS(4/4)\nV:1 o4 v85\n/2 [d f a c'] [g b_ d' f'] | [c e g b_] [f a c' e'] | [b_ d' f' a'] [e g b d'] | [a c' e' g'] [a c' e' g'] |"
}

```

---

## Phase 2: The Fine-Tuning Execution Script

Because you have *billions* of scores, a low-rank parameter-efficient tuning (like a LoRA with a rank of 8) will bottleneck the model's ability to absorb deep music theory. You need a **High-Rank LoRA Matrix** targeting every single linear layer, mimicking a full-parameter rewrite while preserving memory.

We will use the ultra-fast Unsloth ecosystem (which fully supports Gemma 3) to execute this training efficiently.

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 4096 # Gemma 3 supports long contexts, but 4k is perfect for 16-bar multi-track loops
dtype = None # None for auto-detection (Float16/Bfloat16 depending on GPU)
load_in_4bit = False # Use False (Full 16-bit precision) because music syntax requires absolute weight accuracy

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "google/gemma-3-270m",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Deepen LoRA to act as a core domain adaptation matrix
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,          # High rank to capture massive data complexity
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 128,
    lora_dropout = 0.05,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# Resize token embeddings to map our custom MIDIText tokens
model.resize_token_embeddings(len(tokenizer))

# Format Dataset Function
def formatting_prompts_func(examples):
    texts = []
    for inst, inp, out in zip(examples['instruction'], examples['input'], examples['output']):
        # Strict training template
        text = f"### Prompt:\n{inst}\n\n### Parameters:\n{inp}\n\n### Score:\n{out}{tokenizer.eos_token}"
        texts.append(text)
    return { "text" : texts }

dataset = load_dataset("json", data_files="your_processed_miditext_dataset.json")
dataset = dataset.map(formatting_prompts_func, batched = True)

# Training Hyperparameters
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset["train"],
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    packing = False, # Keep individual loops explicitly mapped to single sequences
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.03,
        num_train_epochs = 1, # With billions of tokens, 1 epoch is plenty to prevent overfitting
        learning_rate = 2e-4, # Stable learning rate for High-Rank LoRA
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs",
    ),
)

trainer_output = trainer.train()

```

---

## Phase 3: The Production Deployment Pipeline

Once trained, saving the raw LLM weights is only half the battle. To ensure production-grade outputs that load with **zero syntax breakages** inside a Digital Audio Workstation (DAW), implement a deterministic compilation wrap.

```
[User Generation Prompt]
         ↓
 [Fine-Tuned Gemma 270M]
         ↓
[Regex / BNF Grammar Filter] ➔ (Blocks bad syntax, unclosed brackets, missing bars)
         ↓
 [Valid MIDIText Stream]
         ↓
[Python Lossless Decompiler] ➔ [.MID File Output]
         ↓
   [DAW / Plugin]

```

### 1. Exporting the Weights

For local integration directly inside your converter backend, convert the fine-tuned adapter weights to GGUF format for ultra-low latency execution via a local CPU inference environment:

```python
# Save the model locally as a GGUF file for local DAW/plugin processing
model.save_pretrained_gguf("miditext_genius_gemma", tokenizer, quantization_method = "q8_0")

```

### 2. Enforcing Production Stability with BNF Grammars

Even a beautifully fine-tuned 270M model might occasionally omit a closing bracket `]` when generating a high-velocity chord sequence or miscalculate absolute ticks (`@N`).

When calling the model in your application, force the model using structured generation tools (like **Outlines** or **Guidance**). By passing a Backus-Naur Form (BNF) grammar to the inference engine, you constrain the model's token options at every step:

* If the model opens a chord bracket `[`, it is literally forced to *only* output lowercase notes `a-g`, flats/sharps, and a closing bracket `]`.
* It is programmatically blocked from outputting illegal characters or hallucinating invalid code.

---

## The Ultimate Payoff

By marrying your massive data wealth with a properly structured tokenizer and a high-rank LoRA on Gemma 3 270M, you bypass the need for a bloated 70-billion parameter model. Your resulting local file will weigh less than **400MB**, consume negligible RAM, and instantly output zero-latency, highly creative, syntactically perfect musical notations directly on any standard computer.