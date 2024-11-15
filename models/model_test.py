from transformers import pipeline

model_name = "Butsushi/results"
classifier = pipeline("text-classification", model=model_name)

test_inputs = [
    "I feel like there’s no point in continuing. It might be better if I just disappear.",  # Ideation
    "I've started collecting pills. It feels like it's the only way to stop the pain.",  # Behavior
    "Last night, I tried to overdose, but I couldn't go through with it.",  # Attempt
    "I'm so proud of how far you've come. Keep going, you're doing great!",  # Supportive
    "Ever since my breakup, I haven't been able to sleep, and everything feels pointless."  # Indicator
]

for text in test_inputs:
    result = classifier(text)
    print(f"Input: {text}")
    print(f"Predicted Label: {result[0]['label']}, Confidence Score: {result[0]['score']:.2f}")
    print("-" * 50)