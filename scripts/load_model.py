from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b7")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b7")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, repetition_penalty=1.2, temperature=0.4)