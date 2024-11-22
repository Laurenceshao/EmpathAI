from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, ClassVar

from openCHA.CustomDebugFormatter import CustomDebugFormatter
from openCHA.datapipes import DataPipe
from openCHA.datapipes import DatapipeType
from openCHA.datapipes import initialize_datapipe
from openCHA.llms import LLMType
from openCHA.orchestrator import Action
from openCHA.planners import BasePlanner
from openCHA.planners import initialize_planner
from openCHA.planners import PlanFinish
from openCHA.planners import PlannerType
from openCHA.response_generators import (
    BaseResponseGenerator,
)
from openCHA.response_generators import (
    initialize_response_generator,
)
from openCHA.response_generators import (
    ResponseGeneratorType,
)
# add follow_up_generator
from openCHA.follow_up_generators import (
    BaseFollowUpGenerator,
)
from openCHA.follow_up_generators import (
    initialize_follow_up_generator,
)
from openCHA.follow_up_generators import (
    FollowUpGeneratorType,
)
from openCHA.tasks import BaseTask
from openCHA.tasks import initialize_task
from openCHA.tasks import TaskType
from openCHA.tasks.types import INTERNAL_TASK_TO_CLASS
from openCHA.tasks.internals.suicidal_sensor import SuicidalSensor  # Import the SuicidalSensor Task
from pydantic import BaseModel
from openCHA.vector_database import VectorDatabase
import numpy as np 
import openai 

class Orchestrator(BaseModel):
    """
    **Description:**

        The Orchestrator class is the main execution heart of the CHA. All the components of the Orchestrator are initialized and executed here.
        The Orchestrator will start a new answering cycle by calling the `run` method. From there, the planning is started,
        then tasks will be executed one by one till the **Task Planner** decides that no more information is needed.
        Finally the **Task Planner** final answer will be routed to the **Final Response Generator** to generate an empathic final
        response that is returned to the user.
    """

    planner: BasePlanner = None
    datapipe: DataPipe = None
    promptist: Any = None
    follow_up_generator: BaseFollowUpGenerator = None #new
    response_generator: BaseResponseGenerator = None
    vector_database: VectorDatabase = None #new 
    available_tasks: Dict[str, BaseTask] = {}
    max_retries: int = 5
    max_task_execute_retries: int = 3
    max_planner_execute_retries: int = 16
    max_final_answer_execute_retries: int = 3
    role: int = 0
    verbose: bool = False
    planner_logger: Optional[logging.Logger] = None
    tasks_logger: Optional[logging.Logger] = None
    orchestrator_logger: Optional[logging.Logger] = None
    final_answer_generator_logger: Optional[logging.Logger] = None
    promptist_logger: Optional[logging.Logger] = None
    error_logger: Optional[logging.Logger] = None
    previous_actions: List[str] = []
    current_actions: List[str] = []
    runtime: Dict[str, bool] = {}


    # # Methodology Mapping for Suicide-Risk Levels

    #new mapping
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


    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def print_log(self, log_name: str, message: str):
        if self.verbose:
            if log_name == "planner":
                self.planner_logger.debug(message)
            if log_name == "task":
                self.tasks_logger.debug(message)
            if log_name == "orchestrator":
                self.orchestrator_logger.debug(message)
            if log_name == "response_generator":
                self.final_answer_generator_logger.debug(message)
            if log_name == "promptist":
                self.promptist_logger.debug(message)
            if log_name == "error":
                self.error_logger.debug(message)

    @classmethod
    def initialize(
        self,
        planner_llm: str = LLMType.OPENAI,
        planner_name: str = PlannerType.ZERO_SHOT_REACT_PLANNER,
        datapipe_name: str = DatapipeType.MEMORY,
        promptist_name: str = "",
        response_generator_llm: str = LLMType.OPENAI,
        response_generator_name: str = ResponseGeneratorType.BASE_GENERATOR,
        follow_up_generator_llm: str = LLMType.OPENAI,
        follow_up_generator_name: str = FollowUpGeneratorType.BASE_GENERATOR,
        available_tasks: Optional[List[str]] = None,
        previous_actions: List[Action] = None,
        verbose: bool = False,
        openai_api_key: str = "", #replace with your api key 
        embedding_dim: int = 512,
        categories: List[str] = ["Ideation", "Supportive", "Indicator", "Behavior", "Attempt"],
        rows_per_category: int = 1,
        **kwargs,
    ) -> Orchestrator:
        """
            This class method initializes the Orchestrator by setting up the planner, datapipe, promptist, response generator,
            and available tasks.

        Args:
            planner_llm (str): LLMType to be used as LLM for planner.
            planner_name (str): PlannerType to be used as task planner.
            datapipe_name (str): DatapipeType to be used as data pipe.
            promptist_name (str): Not implemented yet!
            response_generator_llm (str): LLMType to be used as LLM for response generator.
            response_generator_name (str): ResponseGeneratorType to be used as response generator.
            available_tasks (List[str]): List of available task using TaskType.
            previous_actions (List[Action]): List of previous actions.
            verbose (bool): Specifies if the debugging logs be printed or not.
            **kwargs (Any): Additional keyword arguments.
        Return:
            Orchestrator: Initialized Orchestrator instance.


        Example:
            .. code-block:: python

                from openCHA.datapipes import DatapipeType
                from openCHA.planners import PlannerType
                from openCHA.response_generators import ResponseGeneratorType
                from openCHA.tasks import TaskType
                from openCHA.llms import LLMType
                from openCHA.orchestrator import Orchestrator

                orchestrator = Orchestrator.initialize(
                    planner_llm=LLMType.OPENAI,
                    planner_name=PlannerType.ZERO_SHOT_REACT_PLANNER,
                    datapipe_name=DatapipeType.MEMORY,
                    promptist_name="",
                    response_generator_llm=LLMType.OPENAI,
                    response_generator_name=ResponseGeneratorType.BASE_GENERATOR,
                    available_tasks=[TaskType.SERPAPI, TaskType.EXTRACT_TEXT],
                    verbose=self.verbose,
                    **kwargs
                )

        """

        #add internal tasks - remove for testing
        # if available_tasks is None:
        #     print('ADDING INTERNAL TASKS')
        #     available_tasks = INTERNAL_TASK_TO_CLASS.keys()
        # else:
        #     available_tasks += INTERNAL_TASK_TO_CLASS.keys()

        if previous_actions is None:
            previous_actions = []

        planner_logger = (
            tasks_logger
        ) = (
            orchestrator_logger
        ) = (
            final_answer_generator_logger
        ) = promptist_logger = error_logger = None
        if verbose:
            planner_logger = CustomDebugFormatter.create_logger(
                "Planner", "cyan"
            )
            tasks_logger = CustomDebugFormatter.create_logger(
                "Task", "purple"
            )
            orchestrator_logger = CustomDebugFormatter.create_logger(
                "Orchestrator", "green"
            )
            final_answer_generator_logger = (
                CustomDebugFormatter.create_logger(
                    "Response Generator", "blue"
                )
            )
            promptist_logger = CustomDebugFormatter.create_logger(
                "Promptist", "blue"
            )
            error_logger = CustomDebugFormatter.create_logger(
                "Error", "red"
            )

        datapipe = initialize_datapipe(
            datapipe=datapipe_name, **kwargs
        )
        if verbose:
            orchestrator_logger.debug(
                f"Datapipe {datapipe_name} is successfully initialized.\n"
            )

        #initialize databse
        print('initilizing vector database')
        vector_database = VectorDatabase(embedding_dim=embedding_dim, api_key=openai_api_key)

        # print('generating synthetic conversation')
        # # Generate and add synthetic conversations to the vector database
        # synthetic_conversations = self.generate_synthetic_conversations(categories, rows_per_category)
        # for conversation in synthetic_conversations:
        #     text = conversation['Person'] + " " + conversation['Therapist']
        #     embedding = vector_database.generate_embedding(text)
        #     vector_database.add_embeddings(np.array([embedding]), [conversation])

        tasks = {}
        for task in available_tasks:
            kwargs["datapipe"] = datapipe
            tasks[task] = initialize_task(task=task, **kwargs)
            if verbose:
                orchestrator_logger.debug(
                    f"Task '{task}' is successfully initialized."
                )

        self.suicidal_sensor = SuicidalSensor(datapipe=datapipe)  # Initialize the SuicidalSensor task with datapipe

        planner = initialize_planner(
            tasks=list(tasks.values()),
            llm=planner_llm,
            planner=planner_name,
            **kwargs,
        )
        if verbose:
            orchestrator_logger.debug(
                f"Planner {planner_name} is successfully initialized."
            )

        response_generator = initialize_response_generator(
            response_generator=response_generator_name,
            llm=response_generator_llm,
            **kwargs,
        )
        if verbose:
            orchestrator_logger.debug(
                f"Response Generator {response_generator_name} is successfully initialized."
            )

        #add follow up generator
        follow_up_generator = initialize_follow_up_generator(
            follow_up_generator=follow_up_generator_name,
            llm=follow_up_generator_llm,
            **kwargs,
        )
        if verbose:
            orchestrator_logger.debug(
                f"Follow Up Generator {follow_up_generator_name} is successfully initialized."
            )

        return self(
            planner=planner,
            datapipe=datapipe,
            promptist=None,
            response_generator=response_generator,
            follow_up_generator=follow_up_generator,
            vector_database=vector_database,
            available_tasks=tasks,
            verbose=verbose,
            previous_actions=previous_actions,
            current_actions=[],
            planner_logger=planner_logger,
            tasks_logger=tasks_logger,
            orchestrator_logger=orchestrator_logger,
            final_answer_generator_logger=final_answer_generator_logger,
            promptist_logger=promptist_logger,
            error_logger=error_logger,
        )
    
    @staticmethod
    def generate_synthetic_conversations(categories: List[str], rows_per_category: int = 1) -> List[Dict[str, str]]:
        """
        Generate synthetic conversations using OpenAI's API for realistic therapist-patient interactions.

        Args:
            categories (List[str]): List of categories to generate conversations for.
            rows_per_category (int): Number of conversations per category.

        Returns:
            List[Dict[str, str]]: List of synthetic conversations.
        """
        conversations = []
        for category in categories:
            prompt = (
                f"You are tasked with creating synthetic data for mental health interventions, "
                f"focusing on realistic therapist-patient interactions for the '{category}' category. "
                "The conversations should be empathetic, professional, and align with the category's characteristics.\n"
                "Each conversation should include:\n"
                "1. Category: The conversation category.\n"
                "2. Person: A statement from the individual, representing their current state.\n"
                "3. Therapist: A supportive response from the therapist.\n"
                "Avoid explicit or graphic content; instead, focus on empathy, coping strategies, and supportive resources.\n"
                "Generate responses in the following format:\n"
                "Category,Person,Therapist\n"
            )

            for _ in range(rows_per_category):
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Generate a conversation for the '{category}' category."}
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                )
                

                generated_text = response.choices[0].message.content.strip()
                if "\n" in generated_text:
                    lines = generated_text.split("\n")
                    if len(lines) >= 2:
                        person_statement = lines[0].replace("Person:", "").strip()
                        therapist_response = lines[1].replace("Therapist:", "").strip()
                        
                        conversation = {
                            "Category": category,
                            "Person": person_statement,
                            "Therapist": therapist_response
                        }
                        print('appending to conversation: ', conversation)
                        conversations.append(conversation)
        
        return conversations

    def process_meta(self) -> bool:
        """
            This method processes the meta information and returns a boolean value. Currently, it always returns False.

        Return:
            bool: False

        """
        return False

    def _update_runtime(self, action: Action = None):
        if action.output_type:
            self.runtime[action.task_response] = False
        for task_input in action.task_inputs:
            if task_input in self.runtime:
                self.runtime[task_input] = True

    def execute_task(
        self, task_name: str, task_inputs: List[str]
    ) -> Any:
        """
            Execute the specified task based on the planner's selected **Action**. This method executes a specific task based on the provided action.
            It takes an action as input and retrieves the corresponding task from the available tasks dictionary.
            It then executes the task with the given task input. If the task has an output_type, it stores the result in the datapipe and returns
            a message indicating the storage key. Otherwise, it returns the result directly.

        Args:
            task_name (str): The name of the Task.
            task_inputs List(str): The list of the inputs for the task.
        Return:
            str: Result of the task execution.
            bool: If the task result should be directly returned to the user and stop planning.
        """
        self.print_log(
            "task",
            f"---------------\nExecuting task:\nTask Name: {task_name}\nTask Inputs: {task_inputs}\n",
        )
        error_message = ""

        try:
            task = self.available_tasks[task_name]
            result = task.execute(task_inputs)
            self.print_log(
                "task",
                f"Task is executed successfully\nResult: {result}\n---------------\n",
            )
            
            action = Action(
                    task_name=task_name,
                    task_inputs=task_inputs,
                    task_response=result,
                    task_outputs=task.outputs,
                    output_type=task.output_type,
                    datapipe=self.datapipe,
                )

            self._update_runtime(action)

            self.previous_actions.append(action)
            self.current_actions.append(action)
            if task.return_direct:
                print('Returning direct\n')
                #raise ReturnDirectException(result)
            return result  # , task.return_direct
        # except ReturnDirectException as e:
        #     print('Inside return direct except\n')
        #     print('message: ', e.message)
        #     #raise ReturnDirectException(e.message)
        #     raise
        except Exception as e:
            self.print_log(
                "error",
                f"Error running task: \n{e}\n---------------\n",
            )
            logging.exception(e)
            error_message = e
            raise ValueError(
                f"Error executing task {task_name}: {error_message}\n\nTry again with different inputs."
            )

    def planner_generate_prompt(self, query) -> str:
        """
            Generate a prompt from the query to make it more understandable for both planner and response generator.
            Not implemented yet.

        Args:
                query (str): Input query.
        Return:
                str: Generated prompt.

        """
        return query

    def _prepare_planner_response_for_response_generator(self):
        print("runtime", self.runtime)
        final_response = ""
        for action in self.current_actions:
            final_response += action.dict(
                (
                    action.output_type
                    and not self.runtime[action.task_response]
                )
            )
        return final_response

    def generate_metas_prompt(self):
        return "\n\n".join([meta.dict() for meta in self.meta_data])

    def response_generator_generate_prompt(
        self,
        final_response: str = "",
        history: str = "",
        meta: str = "",
        use_history: bool = False,
    ) -> str:
        if meta is None:
            meta = []

        prompt = "MetaData: {meta}\n\nHistory: \n{history}\n\n"
        if use_history:
            prompt = prompt.replace("{history}", history)

        prompt = (
            prompt.replace("{meta}", meta) + f"\n{final_response}"
        )
        return prompt

    def plan(
        self, query, history, meta, use_history, **kwargs
    ) -> str:
        """
            Plan actions based on the query, history, and previous actions using the selected planner type.
            This method generates a plan of actions based on the provided query, history, previous actions, and use_history flag.
            It calls the plan method of the planner and returns a list of actions or plan finishes.

        Args:
            query (str): Input query.
            history (str): History information.
            meta (Any): meta information.
            use_history (bool): Flag indicating whether to use history.
        Return:
            str: A python code block will be returnd to be executed by Task Executor.



        Example:
            .. code-block:: python

                from langchain import ReActChain, OpenAI
                react = ReAct(llm=OpenAI())

        """
        return self.planner.plan(
            query,
            history,
            meta,
            self.previous_actions,
            use_history,
            **kwargs,
        )
    
    def generate_follow_up_questions(self, risk_label: str, query: str) -> List[str]:
        """
        Generate follow-up questions based on risk level instructions using a RAG-based method.

        Args:
            follow_up_instructions (str): Instructions about how to generate follow-up questions.
            query (str): The initial user's statement or question.

        Returns:
            List[str]: A list of follow-up questions.
        """
        # Step 1: Retrieval from Synthetic Data Knowledge Base
        # Generate an embedding for the query
        #query_embedding = self.embedding_generator.generate(query)


        #FOR LATER - implement vector database with synthetic data
        # # Retrieve the top matching conversations from the synthetic data knowledge base
        # retrieved_conversations = self.vector_database.retrieve(query, top_k=3)

        # print('retrieved conversations: ', retrieved_conversations)

        # # Extract retrieved therapist responses for context
        # retrieved_context = ""
        # for conversation in retrieved_conversations:
        #     retrieved_context += f"Category: {conversation['Category']}\n"
        #     retrieved_context += f"Person: {conversation['Person']}\n"
        #     retrieved_context += f"Therapist: {conversation['Therapist']}\n\n"

        #define context as definition of category and overall instructions for now
        if risk_label in self.suicide_risk_mapping:
            risk_methodology = self.suicide_risk_mapping[risk_label]
        else:
            risk_methodology = {
                "instructions": "The user's situation is unclear or does not align with predefined risk categories. Focus on creating a supportive and safe environment. Respond empathetically and encourage open dialogue to better understand their concerns.",
                "methodology": "Empathetic General Support",
                "response_start": "Thank you for reaching out. I want to make sure you feel supported and heard. ",
                "follow_up_instructions": "Ask open-ended, clarifying questions to better understand the user's thoughts or concerns. ",
            }

            
        retrieved_context = risk_methodology

        # Step 2: Define few-shot examples for each category
        few_shot_examples = (
            "\n### Examples of Empathetic Conversations ###\n\n"
            "Category: Supportive\n"
            "Person: 'I really want to make a change in my life, but I keep doubting myself.'\n"
            "Therapist: 'Making changes can be challenging, especially when self-doubt creeps in. Maybe we could talk about some small steps that feel manageable?'\n\n"
            
            "Category: Indicator\n"
            "Person: 'Lately, I feel like I’m just tired of everything, like there’s a constant weight I can’t shake off.'\n"
            "Therapist: 'It sounds like things feel heavy for you right now. I’m here to listen and support you in this journey.'\n\n"
            
            "Category: Ideation\n"
            "Person: 'Sometimes, I wonder if there’s any point to it all.'\n"
            "Therapist: 'When life feels pointless, it can be deeply isolating. I’m here to help you find some hope. Would you feel comfortable sharing more?'\n\n"
            
            "Category: Behavior\n"
            "Person: 'I’ve been taking risks lately, like driving too fast or going places alone at night.'\n"
            "Therapist: 'It sounds like these behaviors might be a way of coping with something deeper. Let’s explore how we can work through this together.'\n\n"
            
            "Category: Attempt\n"
            "Person: 'I tried to end things a few weeks ago. I don’t feel any better.'\n"
            "Therapist: 'I’m so glad you’re here with me now. Talking through these experiences can help, and I’m here to listen and support however I can.'\n"
        )


        prompt_template = (
            "You are an empathetic conversational agent emulating the role of a therapist, engaging in a conversation with someone "
            "identified at the {risk_level} level of suicide risk. Your primary role is to provide empathetic support, encourage open dialogue, "
            "and address their emotional needs based on their identified risk level.\n\n"
            "Risk Level Context:\n{context}\n\n"
            "Instructions for Generating Follow-Up Questions:\n"
            "- Use the user's query to guide the focus of the questions. Dynamically determine the number of follow-up questions needed based on the information gaps in the user's query."
            "- Identify key aspects of the user's mental state that require clarification to better understand their emotional and psychological needs. Each question should address a distinct aspect of the user's mental state or situation, such as triggers, emotional impact, coping mechanisms, or support needs.\n"
            "- Each question should demonstrate empathy, focus on a unique aspect, and encourage open dialogue while validating the user's emotions.\n"
            "- The examples below provide guidance on tone and style but should not dictate the content of your questions.\n\n"
            "Examples of Empathetic Conversations for Each Risk Level:\n{examples}\n\n"
            "User Query:\n{user_query}\n\n"
            "Using the context, instructions, and examples above, generate at least three (more if needed) highly empathetic and supportive follow-up questions that align with the user's input, explore different aspects of their mental state, and encourage open dialogue without being intrusive. Ensure that you always validate the user's feelings and provide emotional reassurance.\n\n"
            "Follow-up questions:"
        )


         # Fill the prompt with the appropriate information
        prompt = prompt_template.format(
            risk_level=risk_label,
            instructions=retrieved_context["follow_up_instructions"],
            context=retrieved_context["instructions"],
            examples=few_shot_examples,
            user_query=query,
        )

        #prompt = prompt_template.format(instructions=follow_up_instructions, context=retrieved_context)

        # Step 3: Use the follow-up generator for generating follow-up questions
        follow_up_questions = self.follow_up_generator.generate(prompt)

        return follow_up_questions

    def generate_final_answer(self, query, thinker, **kwargs) -> str:
        """
            Generate the final answer using the response generator.
            This method generates the final answer based on the provided query and thinker.
            It calls the generate method of the response generator and returns the generated answer.

        Args:
            query (str): Input query.
            thinker (str): Thinking component.
        Return:
            str: Final generated answer.

        """

        retries = 0
        while retries < self.max_final_answer_execute_retries:
            try:
                return self.response_generator.generate(
                    query=query,
                    thinker=thinker,
                    **kwargs,
                )
            except Exception as e:
                logging.exception(e)
                retries += 1
        return "We currently have problem processing your question. Please try again after a while."
    
    def process_plan_result(self, plan_result):
        if not plan_result:
            return "No resources could be retrieved."
        elif isinstance(plan_result, list):
            # Combine the extracted texts into a single string or process as needed
            combined_result = "\n\n".join(plan_result)
            return combined_result
        else:
            return str(plan_result)

    def run(
        self,
        query: str,
        meta: List[str] = None,
        history: str = "",
        use_history: bool = False,
        **kwargs: Any,
    ) -> str:
        """
            This method runs the orchestrator by taking a query, meta information, history, and other optional keyword arguments as input.
            It initializes variables for tracking the execution, generates a prompt based on the query, and sets up a loop for executing actions.
            Within the loop, it plans actions, executes tasks, and updates the previous actions list.
            If a PlanFinish action is encountered, the loop breaks, and the final response is set.
            If any errors occur during execution, the loop retries a limited number of times before setting a final error response.
            Finally, it generates the final response using the prompt and thinker, and returns the final response along with the previous actions.

        Args:
            query (str): Input query.
            meta (List[str]): Meta information.
            history (str): History information.
            use_history (bool): Flag indicating whether to use history.
            **kwargs (Any): Additional keyword arguments.
        Return:
            str: The final response to shown to the user.


        """
        if meta is None:
            meta = []
        i = 0
        meta_infos = ""
        for meta_data in meta:
            key = self.datapipe.store(meta_data)
            meta_infos += (
                f"The file with the name ${meta_data.split('/')[-1]}$ is stored with the key $datapipe:{key}$."
                "Pass this key to the tools when you want to send them over to the tool\n"
            )
        prompt = self.planner_generate_prompt(query)
        if "google_translate" in self.available_tasks:
            prompt = self.available_tasks["google_translate"].execute(
                [prompt, "en"]
            )
            source_language = prompt[1]
            prompt = prompt[0]

        if "CHA" not in history:
            # Step 1: Use SuicidalSensor to assess risk level
            # risk_assessment_result = self.suicidal_sensor.execute([query])
            # print(f"Risk Assessment Result: {risk_assessment_result}")

            # # Step 2: Determine methodology based on risk level
            # risk_label = risk_assessment_result.split(",")[0].split(":")[1].strip()
            # if risk_label in self.suicide_risk_mapping:
            #     risk_methodology = self.suicide_risk_mapping[risk_label]
            # else:
            #     risk_methodology = {
            #         "instructions": "Provide empathetic support.",
            #         "methodology": "General Support",
            #         "response_start": "I want to make sure you feel supported.",
            #         "follow_up_instructions": "Ask clarifying questions to understand the user's concerns better.",
            #     }

            #forced class for now
            risk_assessment_result = (
                "Predicted class: Behavior, Score: 0.85162264108657837, Context: The user exhibits "
                "behaviors indicating a higher suicide risk, such as self-harm or planning. Prioritize "
                "their safety, encourage immediate professional intervention, and assess the need for "
                "emergency services., Risk Relationship: "
                '"The current label is Behavior, meaning the user is exhibiting actions or planning that '
                'suggest a high risk of suicide, such as self-harm or explicit planning. This is a higher '
                "level of risk compared to 'Supportive', 'Indicator', and 'Ideation'. Immediate intervention "
                'is often required to prevent escalation to an actual attempt."'
            )
            risk_label = risk_assessment_result.split(",")[0].split(":")[1].strip()
            if risk_label in self.suicide_risk_mapping:
                risk_methodology = self.suicide_risk_mapping[risk_label]
            else:
                risk_methodology = {
                    "instructions": "Provide empathetic support.",
                    "methodology": "General Support",
                    "response_start": "I want to make sure you feel supported.",
                    "follow_up_instructions": "Ask clarifying questions to understand the user's concerns better.",
                }


            # Step 3: Generate follow-up questions based on risk level
            follow_up_questions = self.generate_follow_up_questions(risk_label, query)
            print(f"Generated Follow-up Questions: {follow_up_questions}")

            # Step 4: Update history with follow-up questions
            #history += f"Follow-up Questions: {follow_up_questions}\n"

            # print('history: ', history)
            # print('end of history')

            #return "\n".join([risk_methodology["response_start"], *follow_up_questions])

            # Remove leading numbers, periods, spaces, and double quotation marks, then format with bullet points
            formatted_questions = "\n".join(
                ["- " + question.lstrip("1234567890. ").strip('"') for question in follow_up_questions]
            )

            # Step 4: Combine the response start with formatted follow-up questions
            return "\n".join([risk_methodology["response_start"], formatted_questions])

        final_response = ""
        finished = False
        self.print_log("planner", "Planning Started...\n")
        while True:
            try:
                self.print_log(
                    "planner",
                    f"Continueing Planning... Try number {i}\n\n",
                )
                actions = self.plan(
                    query=prompt,
                    history=history,
                    meta=meta_infos,
                    use_history=use_history,
                    **kwargs,
                ) 
                # vars = {}
                # exec(actions, locals(), vars)
                # final_response = (
                #     self._prepare_planner_response_for_response_generator()
                # )
                vars = {}
                exec(actions, locals(), vars)
                # After executing, call execute_plan and get the result
                if 'execute_plan' in vars:
                    plan_result = vars['execute_plan']()
                    final_response = self.process_plan_result(plan_result)
                else:
                    raise ValueError("No 'execute_plan' function defined in actions.")

                print("final resp", final_response)
                self.current_actions = []
                self.runtime = {}
                break
            except (Exception, SystemExit) as error:
                self.print_log(
                    "error", f"Planning Error:\n{error}\n\n"
                )
                self.current_actions = []
                i += 1
                if i > self.max_retries:
                    final_response = "Problem preparing the answer. Please try again."
                    break

        self.print_log(
            "planner",
            f"Planner final response: {final_response}\nPlanning Ended...\n\n",
        )

        final_response = self.response_generator_generate_prompt(
            final_response=final_response,
            history=history,
            meta=meta_infos,
            use_history=use_history,
        )

        self.print_log(
            "response_generator",
            f"Final Answer Generation Started...\nInput Prompt: \n\n{final_response}",
        )
        final_response = self.generate_final_answer(
            query=query, thinker=final_response, **kwargs
        )
        self.print_log(
            "response_generator",
            f"Response: {final_response}\n\nFinal Answer Generation Ended.\n",
        )

        if "google_translate" in self.available_tasks:
            final_response = self.available_tasks[
                "google_translate"
            ].execute([final_response, source_language])[0]

        return final_response
