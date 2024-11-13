from typing import Any, List
from openCHA.tasks import BaseTask
from transformers import pipeline
# from openCHA.ask_user import AskUser
import requests

class SuicidalSensor(BaseTask):
    
    name: str = "suicidal_sensor"
    chat_name: str = "SuicidalSensor"
    description: str = "Sense suicidal tendency in texts and give contextual information to openCHA."
    dependencies: List[str] = [] 
    inputs: List[str] = [
        "User input text for processing."
    ]
    outputs: List[str] = []
    output_type: bool = False
    # return_direct: bool = True  
    # api_url: str = "https://api-inference.huggingface.co/models/Butsushi/results"
    # api_token: str = "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"

    def __init__(self, datapipe=None):
        super().__init__(datapipe=datapipe)  # Initialize the parent class with datapipe
        # self.api_url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
        # self.api_token = "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"

    def _execute(self, inputs: List[Any] = None) -> str:
        if inputs is None or len(inputs) == 0:
            return "No input provided."
        
        classifier = pipeline("text-classification", model="Butsushi/results")
        
        user_input = inputs[0]

        # headers = {
        #     "Authorization": f"Bearer {self.api_token}",
        #     "Content-Type": "application/json"
        # }

        # data = {
        #     "inputs": user_input
        # }
        
        try:

            
            # response = requests.post(self.api_url, headers=headers, json=data)
            # response.raise_for_status()
            # api_result = response.json()

            api_result = classifier(user_input)

            print(api_result)
            
            if isinstance(api_result, list) and len(api_result) > 0:
                class_probabilities = api_result[0]['score']
                predicted_class = api_result[0]['label']
                result = f"Predicted class: {predicted_class}, Scores: {class_probabilities}"
            else:
                result = "No valid response from the model."
        except requests.exceptions.RequestException as e:
            return f"Error contacting Hugging Face model API: {e}"
        
        return result


    def explain(self) -> str:
        """Explain what this task does in detail."""
        return "This task processes the user input with MyModel and provides a response."