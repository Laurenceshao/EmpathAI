from __future__ import annotations

from typing import Any, List
from openCHA.llms import BaseLLM
from pydantic import BaseModel
import openai  # Assuming we're using OpenAI's API for LLM-based follow-up question generation

class BaseFollowUpGenerator(BaseModel):
    """
    **Description:**

        Base class for a follow-up generator, providing a foundation for generating follow-up questions using a language model.
    """

    llm_model: BaseLLM = None
    prefix: str = ""
    max_tokens_allowed: int = 1000

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _follow_up_generator_type(self):
        return "base"

    @property
    def _follow_up_generator_model(self):
        return self.llm_model

    @property
    def _generator_prompt(self):
        return (
            "You are an empathetic conversational agent tasked with generating appropriate follow-up questions "
            "for someone expressing a specific level of suicidal risk. Please generate three highly empathetic "
            "and supportive follow-up questions based on the instructions provided below. Ensure that the questions "
            "encourage dialogue without being intrusive.\n"
            "Instructions: {instructions}\n"
            "Follow-up questions:"
        )

    def divide_text_into_chunks(
        self,
        input_text: str = "",
        max_tokens: int = 1000,
    ) -> List[str]:
        """
        Divide the input text into smaller chunks to fit within token limits.

        Args:
            input_text (str): the input text (e.g., prompt).
            max_tokens (int): Maximum number of tokens allowed.
        Return:
            chunks(List): List of string variables
        """
        # 1 token ~= 4 chars in English
        chunks = [
            input_text[i : i + max_tokens * 4]
            for i in range(0, len(input_text), max_tokens * 4)
        ]
        return chunks

    def generate(
        self,
        instructions: str = "",
        **kwargs: Any,
    ) -> List[str]:
        """
        Generate follow-up questions based on the input instructions.

        Args:
            instructions (str): Instructions about how to generate follow-up questions.
            **kwargs (Any): Additional keyword arguments.
        Return:
            List[str]: A list of generated follow-up questions.
        """
        #prompt = self._generator_prompt.replace("{instructions}", instructions)
        prompt = instructions 
        print('follow-up generator prompt: ', prompt)
        response = self._follow_up_generator_model.generate(
            query=prompt,
            max_tokens=self.max_tokens_allowed,
            **kwargs
        )

        # Extract the generated follow-up questions from the response
        follow_up_questions = response.strip().split('\n')
        # Clean up any empty or malformed entries
        follow_up_questions = [q.strip() for q in follow_up_questions if q.strip()]

        return follow_up_questions[:3]  # Return only the first three questions
