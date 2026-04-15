# testmodel.py - tests only your local trained model
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

test_text = """
In the modern digital era, people are often faced with a large amount of text
from articles, reports, and online content, making it difficult to quickly
identify and extract important information. Reading lengthy documents consumes
significant time and may lead to missing critical insights. Existing text
summarization tools generate fixed-length summaries and often act as black boxes,
providing no explanation for why specific sentences are included, which limits
trust and transparency. To address these challenges, this project proposes an
AI-based Explainable Text Summarization System that integrates pre-trained
Transformer models with user-controlled summary length and explainable outputs.
"""

tokenizer = BartTokenizer.from_pretrained("my_bart_model")
model = BartForConditionalGeneration.from_pretrained("my_bart_model")
model.eval()

inputs = tokenizer(test_text, return_tensors="pt", max_length=512, truncation=True)

print("Testing num_beams=4...")
with torch.no_grad():
    out = model.generate(
        inputs["input_ids"],
        num_beams=4,
        max_length=130,
        min_length=50,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )
print("BEAMS=4:", tokenizer.decode(out[0], skip_special_tokens=True))

print()
print("Testing num_beams=1 (greedy)...")
with torch.no_grad():
    out = model.generate(
        inputs["input_ids"],
        num_beams=1,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        max_length=130,
        min_length=50,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5
    )
print("GREEDY+SAMPLE:", tokenizer.decode(out[0], skip_special_tokens=True))