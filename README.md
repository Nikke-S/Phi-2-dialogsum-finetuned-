# Phi-2 Dialogue Summarization

This repository contains a **fine-tuned version of Phi-2**, optimized for **dialogue summarization**. The model takes conversations as input and generates concise summaries.

## Features
- Fine-tuned using **PEFT (Parameter-Efficient Fine-Tuning)**
- Supports summarization of **dialogues, chat logs, and transcripts**
- Uses **Hugging Face `transformers` library** for inference

## Repository Structure
- `fine_tuning_phi_2_nikke.ipynb` → Jupyter Notebook for fine-tuning
- `README.md` → This file (model description and usage)

## Setup
Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

## Training
Run the fine-tuning notebook:
```bash
jupyter notebook fine_tuning_phi_2_nikke.ipynb
```

## Inference (Generating Summaries)
```bash
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-username/phi-2-dialogue-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Summarize the following conversation:\n\n#Person1#: Hello! How are you?\n#Person2#: I'm good, thanks. How about you?\n\nSummary:"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = model.generate(input_ids, max_length=100)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## License
This project is released under the MIT License.

## Notes
The model is available on Hugging Face https://huggingface.co/NikkeS/Phi-2-dialogsum-finetuned
