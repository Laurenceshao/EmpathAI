from typing import Any, List
from openCHA.tasks.task import BaseTask
from transformers import pipeline
import requests

class SuicidalSensor(BaseTask):
    
    name: str = "suicidal_sensor"
    chat_name: str = "SuicidalSensor"
    description: str = "Sense suicidal tendency in texts and provide contextual information to openCHA, including risk level and supportive context."
    dependencies: List[str] = [] 
    inputs: List[str] = [
        "User input text for processing."
    ]
    outputs: List[str] = [
        "Predicted class of suicidal risk.",
        "Confidence score for the prediction.",
        "Detailed contextual information about the risk level."
    ]
    output_type: bool = False

    def __init__(self, datapipe=None):
        super().__init__(datapipe=datapipe)  # Initialize the parent class with datapipe

    def _execute(self, inputs: List[Any] = None) -> str:
        if inputs is None or len(inputs) == 0:
            return "No input provided."
        
        classifier = pipeline("text-classification", model="Butsushi/results")
        
        user_input = inputs[0]
        
        try:
            api_result = classifier(user_input)

            if isinstance(api_result, list) and len(api_result) > 0:
                class_probabilities = api_result[0]['score']
                predicted_class_label = api_result[0]['label']

                # Map the predicted label to the appropriate category
                label_mapping = {
                    "LABEL_0": "Supportive",
                    "LABEL_1": "Indicator",
                    "LABEL_2": "Ideation",
                    "LABEL_3": "Behavior",
                    "LABEL_4": "Attempt"
                }
                predicted_class = label_mapping.get(predicted_class_label, "Unknown")
                
                # Adding more context based on the predicted class, including risk level relationships
                context_info = self._get_context_info(predicted_class)
                risk_relationship = self._get_risk_relationship_prompt(predicted_class)
                result = f"Predicted class: {predicted_class}, Score: {class_probabilities}, Context: {context_info}, Risk Relationship: {risk_relationship}"
            else:
                result = "No valid response from the model."
        except requests.exceptions.RequestException as e:
            return f"Error contacting Hugging Face model API: {e}"
        
        print('Suicide sensor result: ', result)
        return result

    def _get_context_info(self, predicted_class: str) -> str:
        """
        Provide detailed contextual information based on the predicted class.
        """
        context_mapping = {
            "Supportive": "The user is engaging in supportive behavior, offering help or empathy without showing signs of suicidal ideation or behavior.",
            "Indicator": "The user is using language that suggests risk factors, such as discussing a history of mental health issues or personal loss, but not explicitly expressing suicidal thoughts.",
            "Ideation": "The user is expressing thoughts of suicide, indicating a level of ideation that requires attention.",
            "Behavior": "The user is exhibiting behavior or planning actions that suggest a high risk of suicide, such as self-harm or explicit planning.",
            "Attempt": "The user has either attempted suicide or is actively planning an attempt, indicating an urgent need for intervention."
        }
        return context_mapping.get(predicted_class, "No additional context available.")

    def _get_risk_relationship_prompt(self, predicted_class: str) -> str:
        """
        Provide information about the risk level of the current label in relation to other labels via a textual prompt.
        """
        risk_relationship_prompts = {
            "Supportive": (
                "The current label is 'Supportive', which indicates the lowest level of risk. Users in this category are offering support or empathy "
                "without expressing suicidal ideation or behavior. Compared to other risk levels, 'Supportive' users are not showing any active signs of risk, "
                "making this the least severe category."
            ),
            "Indicator": (
                "The current label is 'Indicator', which suggests some risk factors are present, such as discussions of mental health issues or personal loss. "
                "However, there are no explicit signs of suicidal ideation or behavior. This level is higher than 'Supportive' but still lower than 'Ideation', "
                "'Behavior', or 'Attempt'. Monitoring is recommended, but immediate intervention may not be necessary."
            ),
            "Ideation": (
                "The current label is 'Ideation', indicating that the user is expressing suicidal thoughts. This level of risk is higher than both 'Supportive' and 'Indicator' "
                "and requires attention. It is still lower than 'Behavior' or 'Attempt', which involve more concrete actions or planning. Intervention may be needed depending "
                "on the severity of the ideation."
            ),
            "Behavior": (
                "The current label is 'Behavior', meaning the user is exhibiting actions or planning that suggest a high risk of suicide, such as self-harm or explicit planning. "
                "This is a higher level of risk compared to 'Supportive', 'Indicator', and 'Ideation'. Immediate intervention is often required to prevent escalation to an actual attempt."
            ),
            "Attempt": (
                "The current label is 'Attempt', indicating that the user has either attempted suicide or is actively planning an attempt. This is the highest level of risk "
                "and requires urgent intervention. Compared to all other categories, 'Attempt' represents the most severe and immediate danger to the user's safety."
            )
        }
        return risk_relationship_prompts.get(predicted_class, "No information available on risk relationships.")

    def explain(self) -> str:
        """Explain what this task does in detail."""
        return (
            "This task processes the user input using a fine-tuned model to detect suicidal tendencies. "
            "It predicts the risk level (Supportive, Indicator, Ideation, Behavior, Attempt) and provides contextual information "
            "to help the agent framework understand the user's mental state and respond appropriately. "
            "Additionally, it provides information on the relative risk level of the current label in comparison to other labels using a detailed textual prompt, "
            "allowing the agent to better assess the urgency and appropriate response needed."
        )
