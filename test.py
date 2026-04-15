from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ==============================
# DEVICE
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")

# ==============================
# USER INPUT
# ==============================

print("Enter your text (Press Enter twice to finish):\n")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

text = " ".join(lines)

if len(text.strip()) == 0:
    print("No text entered!")
    exit()

# ==============================
# MODEL SELECTION
# ==============================

print("\nSelect Model:")
print("1 - BART")
print("2 - T5")

choice = input("Enter choice (1 or 2): ")

# ==============================
# SUMMARY TYPE
# ==============================

print("\nSelect Summary Length:")
print("1 - Small")
print("2 - Medium")
print("3 - Detailed")

length_choice = input("Enter choice (1, 2 or 3): ")

if length_choice == "1":
    max_len = 40
elif length_choice == "2":
    max_len = 80
elif length_choice == "3":
    max_len = 150
else:
    print("Invalid choice! Defaulting to Medium.")
    max_len = 80

min_len = int(max_len * 0.4)

# ==============================
# LOAD SELECTED MODEL ONLY
# ==============================

print("\nLoading model... Please wait.\n")

if choice == "1":
    tokenizer = AutoTokenizer.from_pretrained("my_bart_model")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "my_bart_model"
        # You can add low_cpu_mem_usage=True if needed
    ).to(device)
elif choice == "2":
    tokenizer = AutoTokenizer.from_pretrained("my_t5_model")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "my_t5_model"
    ).to(device)
    text = "summarize: " + text
else:
    print("Invalid model choice!")
    exit()

print("Model loaded successfully!\n")

# ==============================
# GENERATE SUMMARY
# ==============================

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=1024  # handles longer text safely
).to(device)

# Settings
do_sample_flag = False if length_choice == "3" else True  # Detailed = deterministic
summary_ids = model.generate(
    inputs["input_ids"],
    max_length=max_len,
    min_length=min_len,
    num_beams=6 if do_sample_flag==False else 1,
    length_penalty=2.0 if do_sample_flag==False else 1.0,
    do_sample=do_sample_flag,
    no_repeat_ngram_size=3,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==============================
# OUTPUT
# ==============================

print("\n==============================")
print("GENERATED SUMMARY:\n")
print(summary)
print("\nSummarization completed successfully!")