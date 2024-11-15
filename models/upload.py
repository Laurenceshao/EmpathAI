from huggingface_hub import login
from transformers import AutoModelForSequenceClassification, AutoTokenizer

login()

model_path = "output"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.push_to_hub("suicidal_sensor_class_weighted")
tokenizer.push_to_hub("suicidal_sensor_class_weighted")