"""
Heavily borrowed from langchain: https://github.com/langchain-ai/langchain/
"""
import re
from typing import Any, Dict, List, ClassVar, Optional

from openCHA.planners import Action
from openCHA.planners import BasePlanner
from openCHA.planners import PlanFinish


class EmpatheticTreeOfThoughtPlanner(BasePlanner):
    """
    **Description:**

        This class implements Tree of Thought planner, which inherits from the BasePlanner base class.
        Tree of Thought employs parallel chain of thoughts startegies and decides which one is more
        suitable to proceed to get to the final answer.
        `Paper <https://arxiv.org/abs/2305.10601>`_

        This code defines a base class called "BasePlanner" that inherits from the "BaseModel" class of the pydantic library.
        The BasePlanner class serves as a base for implementing specific planners.


    """

    suicide_risk_mapping: ClassVar[Dict[str, Dict[str, str]]] = {
        "Ideation": {
            "instructions": "The user is expressing suicidal thoughts or ideation. Respond with empathy, validate their feelings, and gently encourage seeking professional help.",
            "methodology": "Empathetic Engagement and Professional Referral",
            "response_start": "I'm so sorry to hear you're feeling this way. Thank you for sharing—it takes courage. I'm here to support you:",
            "follow_up_instructions": "Ask open-ended questions to explore the user's emotional state and any specific triggers or stressors. Example: 'Can you share more about what's been on your mind recently?'"
        },
        "Behavior": {
            "instructions": "The user exhibits behaviors indicating a higher suicide risk, such as self-harm or planning. Prioritize their safety, encourage immediate professional intervention, and assess the need for emergency services.",
            "methodology": "Urgent Risk Management and Safety Planning",
            "response_start": "I can sense that you're in significant distress, and your safety is my top priority:",
            "follow_up_instructions": "Explore the nature of the behavior or planning to assess severity. Confirm if the user is in immediate danger and suggest contacting emergency services. Example: 'Have you acted on any of these thoughts, or do you have a plan in mind?'"
        },
        "Attempt": {
            "instructions": "The user has made or is planning a suicide attempt. Respond with urgency, prioritize their immediate safety, and strongly recommend emergency services.",
            "methodology": "Crisis Intervention and Emergency Protocol",
            "response_start": "This sounds very serious, and I’m here to ensure you’re safe. Please, reach out to emergency services immediately or let me help you connect with someone who can support you right now:",
            "follow_up_instructions": "Directly assess the user's current situation. Example: 'Are you safe right now?' Offer support to connect them with crisis services and encourage involving someone they trust."
        },
        "Indicator": {
            "instructions": "The user mentions risk factors (e.g., personal loss or isolation) but without explicit suicidal thoughts. Provide emotional validation, explore potential risks, and offer professional resources.",
            "methodology": "Proactive Support and Monitoring",
            "response_start": "Thank you for sharing this with me. I can see how challenging this must be for you:",
            "follow_up_instructions": "Ask about the user's support network or coping strategies. Example: 'Have you been able to talk to anyone about how you’re feeling?' or 'What kind of support do you feel might help you right now?'"
        },
        "Supportive": {
            "instructions": "The user offers support to others without expressing personal risk. Acknowledge their empathy, provide general resources, and encourage ongoing dialogue.",
            "methodology": "Community Engagement and Positive Reinforcement",
            "response_start": "Thank you for offering support—it’s so meaningful and impactful to have someone like you caring for others:",
            "follow_up_instructions": "Encourage the user to continue sharing their thoughts or experiences. Example: 'Is there anything you’d like to discuss further or share about your own experiences?'"
        }
    }

    summarize_prompt: bool = True
    max_tokens_allowed: int = 10000

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _planner_type(self):
        return "zero-shot-react-planner"

    @property
    def _planner_model(self):
        return self.llm_model

    @property
    def _response_generator_model(self):
        return self.llm_model

    @property
    def _stop(self) -> List[str]:
        return ["Wait"]

    @property
    def _shorten_prompt(self):
        return (
            "Summarize the following text. Make sure to keep the main ideas "
            "and objectives in the summary. Keep the links "
            "exactly as they are: "
            "{chunk}"
        )

    @property
    def _planner_prompt(self):
            return [
    #         """You are a knowledgeable and empathetic health assistant, tasked with providing tailored support based on a user's identified suicide risk level. 
    # Their risk level has been assessed as: {risk_label}. Use this context to guide your planning.

    # Risk Level Instructions:
    # {instructions}

    # Methodology Context:
    # {methodology}

    # Here are the tools available for planning:
    # {tool_names}

    # The following is the format of the information provided:
    # MetaData: Contains names of data files (e.g., text, audio). Use these files with the tools as needed.
    # History: Contains previous conversations. Use this for continuity in dialogue.
    # PreviousActions: Details of actions already performed.

    # USER QUERY: {input}

    # Develop three strategies to address the user’s query. Each strategy should:
    # 1. Respect the user's emotional state and prioritize safety.
    # 2. Use available tools effectively to gather necessary context.
    # 3. Propose empathetic follow-up steps.

    # Explain the pros and cons of each strategy, then select the best one for implementation.

    # Start the final decision with:
    # 'Decision:'.

    # MetaData:
    # {meta}
    # =========================
    # {previous_actions}
    # =========================
    # {history}
    # =========================
    # USER QUERY: {input}\n"""
    """You are a knowledgeable and empathetic health assistant, tasked with providing tailored support based on a user's identified suicide risk level.
    Their risk level has been assessed as: {risk_label}. Use this context to guide your planning.

    Risk Level Instructions:
    {instructions}

    Methodology Context:
    {methodology}

    Quantity Factor:
    Based on the user's risk level, you must generate a **list of {quantity_factor} resource queries** that address their needs effectively. These queries will be used to retrieve resources for the user.

    ### Task:
    1. Generate {quantity_factor} distinct queries for the `serpapi` tool. These queries should be:
    - Relevant to the user's risk level and emotional state.
    - Prioritized based on urgency and usefulness.
    - Specific enough to yield actionable results.

    ### Examples:
    - For a user at the 'Behavior' risk level (quantity_factor = 5):
    - Queries: ["nearest crisis hotline", "local mental health clinic", "emergency counseling services", "suicide prevention resources", "safety planning templates"]
    - For a user at the 'Ideation' risk level (quantity_factor = 3):
    - Queries: ["coping mechanisms for suicidal ideation", "online therapy options", "self-help guides for managing emotions"]
    - For a user at the 'Supportive' risk level (quantity_factor = 1):
    - Queries: ["general mental health resources"]

    Generate a **list of queries** based on the user's risk level and quantity_factor.    
    # Start the final response with: 'Decision:'.

    MetaData:
    {meta}
    =========================
    {previous_actions}
    =========================
    {history}
    =========================
    USER QUERY: {input}\n"""
    ,
    # """{strategy}
    #     =========================
    #     {previous_actions}
    #     =========================
    #     Tools:
    #     {tool_names}
    #     =========================

    #     You are a skilled Python programmer. Using the selected final strategy mentioned in 'Decision:', write Python code that:

    #     1. Uses the `serpapi` tool to search for relevant resources.
    #     - Example: `search_result = self.execute_task('serpapi', ['query'])`
    #     - Validate that the result contains a key `'url'`. If not, handle it gracefully.

    #     2. Uses the `extract_text` tool to extract content from the retrieved URL.
    #     - Example: `extracted_text = self.execute_task('extract_text', [url])`
    #     - Validate that the result is not empty or invalid. If the result indicates an error, stop execution and log the issue.

    #     3. Chains these tasks together, ensuring that the output of `serpapi` is used as the input to `extract_text`.

    #     Include explicit validation steps for tool outputs and handle errors gracefully. Ensure the generated code uses this structure:
    #     ```python
    #     search_result = self.execute_task('serpapi', ['query'])
    #     if 'error' in search_result or 'url' not in search_result:
    #         raise ValueError(f"Search failed: {search_result.get('error', 'No URL found')}")

    #     extracted_text = self.execute_task('extract_text', [search_result['url']])
    #     if 'error' in extracted_text:
    #         raise ValueError(f"Text extraction failed: {extracted_text['error']}")
    #     ```

    #     Write the complete code inside a Python code block.
    #     """

    """Decision:
    {strategy}
    =========================
    Tools:
    {tool_names}
    =========================
    Using the list of queries generated in the first part, write Python code to execute the following plan:
    1. Use the `serpapi` tool to search for resources using each query.
    - Example: `search_result = self.execute_task('serpapi', [query])`
    - Validate that each result contains a key `'url'`. If not, handle it gracefully.
    2. Use the `extract_text` tool to summarize the content of each retrieved resource.
    - Example: `extracted_text = self.execute_task('extract_text', [url])`
    - Validate that the result is not empty or invalid. If the result indicates an error, skip it.
    3. Chain these tasks together, ensuring that the outputs of `serpapi` are used as inputs to `extract_text`.
    4. Aggregate all successfully retrieved and summarized resources into a list.

    Write Python code in the following structure:
    ```python
    def execute_plan():
        results = []
        queries = {strategy}  # Replace with the generated list of queries

        for query in queries:
            # Step 1: Use SerpAPI to search for resources
            search_result = self.execute_task('serpapi', [query])
            if 'error' in search_result or 'url' not in search_result:
                continue  # Skip if invalid

            # Step 2: Use ExtractText to summarize the resource
            extracted_text = self.execute_task('extract_text', [search_result['url']])
            if 'error' not in extracted_text:
                results.append(extracted_text)

        # Return aggregated results
        return results

            ```

        Write the complete code inside a Python code block.
        """
    # """{strategy}
    #     =========================
    #     {previous_actions}
    #     =========================
    #     Tools:
    #     {tool_names}
    #     =========================

    #     You are a skilled Python programmer. Using the selected final strategy mentioned in 'Decision:', write Python code that:

    #     1. Defines a function `def execute_plan():` to encapsulate the logic.
    #     2. Uses the `quantity_factor` variable to determine how many resources to fetch.
    #        - For example, use a loop to make `quantity_factor` calls to `serpapi` or adjust the search to retrieve multiple results at once.
    #     3. Uses the `serpapi` tool to search for relevant resources.
    #        - Example: `search_result = self.execute_task('serpapi', ['query'])`
    #        - Validate that each result contains a key `'url'`. If not, handle it gracefully.
    #     4. Uses the `extract_text` tool to extract content from each retrieved URL.
    #        - Example: `extracted_text = self.execute_task('extract_text', [url])`
    #        - Validate that the result is not empty or invalid. If the result indicates an error, handle it appropriately.
    #     5. Chains these tasks together, ensuring that the outputs of `serpapi` are used as inputs to `extract_text`.
    #     6. Aggregates the results from all successful extractions.

    #     Include explicit validation steps for tool outputs and handle errors gracefully. Ensure the generated code uses this structure:

    #     ```python
    #     def execute_plan():
    #         results = []
    #         for i in range(quantity_factor):
    #             search_result = self.execute_task('serpapi', ['query'])
    #             if 'error' in search_result or 'url' not in search_result:
    #                 continue  # Skip if invalid

    #             extracted_text = self.execute_task('extract_text', [search_result['url']])
    #             if 'error' not in extracted_text:
    #                 results.append(extracted_text)

    #         return results
    #     ```

    #     Write the complete code inside a Python code block.
    #     """
    ]

    def task_descriptions(self):
        return "".join(
            [
                (
                    "\n-----------------------------------\n"
                    f"**{task.name}**: {task.description}"
                    "\nThis tool have the following outputs:\n"
                    + "\n".join(task.outputs)
                    + (
                        "\n- The result of this tool will be stored in the datapipe."
                        if task.output_type
                        else ""
                    )
                    + "\n-----------------------------------\n"
                )
                for task in self.available_tasks
            ]
        )

    def divide_text_into_chunks(
        self,
        input_text: str = "",
        max_tokens: int = 10000,
    ) -> List[str]:
        """
        Generate a response based on the input prefix, query, and thinker (task planner).

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

    def generate_scratch_pad(
        self, previous_actions: List[str] = None, **kwargs: Any
    ):
        if previous_actions is None:
            previous_actions = []

        agent_scratchpad = ""
        if len(previous_actions) > 0:
            agent_scratchpad = "\n".join(
                [f"\n{action}" for action in previous_actions]
            )
        # agent_scratchpad
        if (
            self.summarize_prompt
            and len(agent_scratchpad) / 4 > self.max_tokens_allowed
        ):
            # Shorten agent_scratchpad
            chunks = self.divide_text_into_chunks(
                input_text=agent_scratchpad,
                max_tokens=self.max_tokens_allowed,
            )
            agent_scratchpad = ""
            kwargs["max_tokens"] = min(
                2000, int(self.max_tokens_allowed / len(chunks))
            )
            for chunk in chunks:
                prompt = self._shorten_prompt.replace(
                    "{chunk}", chunk
                )
                chunk_summary = (
                    self._response_generator_model.generate(
                        query=prompt, **kwargs
                    )
                )
                agent_scratchpad += chunk_summary + " "

    def plan(
        self,
        query: str,
        history: str = "",
        meta: str = "",
        previous_actions: List[str] = None,
        use_history: bool = False,
        risk_label: str = "Ideation",  # Default risk level
        **kwargs: Any,
    ) -> str:
        """
            Generate a plan using Empathetic Tree of Thought

        Args:
            query (str): Input query.
            history (str): History information.
            meta (str): meta information.
            previous_actions (List[Action]): List of previous actions.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            Action: return action.

        """
        if previous_actions is None:
            previous_actions = []

        previous_actions_prompt = ""
        if len(previous_actions) > 0 and self.use_previous_action:
            previous_actions_prompt = f"Previoius Actions:\n{self.generate_scratch_pad(previous_actions, **kwargs)}"


        # Retrieve methodology for the risk label
        risk_context = self.suicide_risk_mapping.get(
            risk_label,
            {
                "instructions": "Provide empathetic support and assess user's emotional state.",
                "methodology": "General Emphathetic Support",
            },
        )

        # Define the quantity factor based on the risk level
        quantity_factor = {
            "Supportive": 1,  # Minimal resources for supportive conversations
            "Indicator": 2,  # Moderate resources for users with risk indicators
            "Ideation": 3,  # More resources for users expressing ideation
            "Behavior": 5,  # Urgent: Provide a wide range of immediate resources
            "Attempt": 5,  # Emergency: Ensure multiple options for assistance
        }.get(risk_label, 1)

        # Adjust instructions for tool usage based on risk level
        if risk_label in ["Behavior", "Attempt"]:
            tool_instructions = (
                f"Use the SerpAPI tool to search for at least **{quantity_factor} emergency resources**, including local crisis hotlines, "
                "mental health clinics, and emergency response services. Follow up by using the ExtractText tool to summarize "
                "the information from the webpages."
            )
        elif risk_label in ["Ideation", "Indicator"]:
            tool_instructions = (
                f"Use the SerpAPI tool to search for **{quantity_factor} self-help resources**, therapy options, and coping strategies. "
                "Provide detailed summaries using the ExtractText tool."
            )
        else:
            tool_instructions = (
                "Use the SerpAPI tool sparingly to search for general mental health resources or informational articles."
            )

        # Merge tool instructions with the risk methodology
        risk_context["instructions"] += f"\n\n{tool_instructions}"

        # # Inject risk context into the planner prompt
        # prompt = (
        #     self._planner_prompt[0]
        #     .replace("{input}", query)
        #     .replace("{meta}", meta)
        #     .replace(
        #         "{history}", history if use_history else "No History"
        #     )
        #     .replace("{previous_actions}", "\n".join(previous_actions))
        #     .replace("{tool_names}", self.task_descriptions())
        #     .replace("{risk_label}", risk_label)
        #     .replace("{instructions}", risk_context["instructions"])
        #     .replace("{methodology}", risk_context["methodology"])
        # )

        # Inject risk context into the planner prompt
        prompt = (
            self._planner_prompt[0]
            .replace("{input}", query)
            .replace("{meta}", meta)
            .replace("{history}", history if use_history else "No History")
            .replace("{previous_actions}", "\n".join(previous_actions))
            .replace("{tool_names}", self.task_descriptions())
            .replace("{risk_label}", risk_label)
            .replace("{instructions}", risk_context["instructions"])
            .replace("{methodology}", risk_context["methodology"])
        )


        
        print(prompt)
        kwargs["max_tokens"] = 1000
        response = self._planner_model.generate(
            query=prompt, **kwargs
        )
        print("respp\n\n", response)
        # prompt = (
        #     self._planner_prompt[1]
        #     .replace(
        #         "{strategy}",
        #         "Decision:\n" + response.split("Decision:")[-1],
        #     )
        #     .replace("{tool_names}", self.get_available_tasks())
        #     .replace("{previous_actions}", previous_actions_prompt)
        #     .replace("{input}", query)
        # )

        prompt = (
            self._planner_prompt[1]
            .replace("{strategy}", "Decision:\n" + response.split("Decision:")[-1])
            .replace("{tool_names}", self.task_descriptions())
            .replace("{previous_actions}", "\n".join(previous_actions))
            .replace("{input}", query)
        )

        print("prompt2\n\n", prompt)
        kwargs["stop"] = self._stop
        response = self._planner_model.generate(
            query=prompt, **kwargs
        )

        index = min([response.find(text) for text in self._stop])
        response = response[0:index]
        actions = self.parse(response)
        print("actions", actions)
        return actions

    def parse(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """
            Parse the output query into a list of actions or a final answer. It parses the output based on \
            the following format:

                Action: action\n
                Action Inputs: inputs

        Args:\n
            query (str): The planner output query to extract actions.
            **kwargs (Any): Additional keyword arguments.
        Return:
            List[Union[Action, PlanFinish]]: List of parsed actions or a finishing signal.
        Raise:
            ValueError: If parsing encounters an invalid format or unexpected content.

        """
        pattern = r"`+python\n(.*?)`+"
        code = re.search(pattern, query, re.DOTALL).group(1)
        return code
