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


    # Methodology Mapping for Suicide-Risk Levels
    suicide_risk_mapping: ClassVar[Dict[str, Dict[str, str]]] = {
        "Ideation": {
            "instructions": "The user is expressing suicidal thoughts or ideation. Provide empathetic support and recommend professional help if appropriate.",
            "methodology": "Supportive Counseling and Referral",
            "response_start": "Thank you for sharing this. I hear that you're having thoughts about suicide, and I'm here to help:",
            "follow_up_instructions": "Generate follow-up questions to understand the user's emotional state and any specific triggers for their thoughts.",
        },
        "Behavior": {
            "instructions": "The user has indicated behaviors consistent with a higher risk of suicide, such as self-harm or planning. Address the immediate risk and suggest contacting emergency services or professionals.",
            "methodology": "Risk Escalation Management",
            "response_start": "It sounds like you're in significant distress, and I'm very concerned for your safety:",
            "follow_up_instructions": "Generate follow-up questions to assess the immediate severity of risk and determine if urgent intervention is necessary.",
        },
        "Attempt": {
            "instructions": "The user has made or is planning a suicide attempt. Provide a crisis response, and encourage contacting emergency services immediately.",
            "methodology": "Crisis Response Protocol",
            "response_start": "This is very serious, and I want to make sure you’re safe. Please contact emergency services immediately:",
            "follow_up_instructions": "Ask if the user is safe now and if they have access to emergency support. Offer ways to reach professional help.",
        },
        "Indicator": {
            "instructions": "The user has mentioned risk factors, such as personal loss, without explicit suicidal thoughts. Provide supportive responses and suggest professional resources.",
            "methodology": "Risk Indicator Monitoring",
            "response_start": "Thank you for sharing. I understand that these experiences can be really tough:",
            "follow_up_instructions": "Generate follow-up questions to understand the user's support network and other contributing factors.",
        },
        "Supportive": {
            "instructions": "The user is offering support to others without expressing personal risk. Provide general resources and continue the supportive discussion.",
            "methodology": "Community Support Facilitation",
            "response_start": "Thank you for offering support. It’s so important that we support each other:",
            "follow_up_instructions": "Ask if there’s anything else the user would like to discuss or contribute.",
        },
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
        openai_api_key: str = "sk-proj-DqqRsFRfH0o2tjZUDyfCT3BlbkFJ12mQk0UjEmXcTx3jgahe",
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
        #add internal tasks 
        if available_tasks is None:
            print('ADDING INTERNAL TASKS')
            available_tasks = INTERNAL_TASK_TO_CLASS.keys()
        else:
            available_tasks += INTERNAL_TASK_TO_CLASS.keys()

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

        print('generating synthetic conversation')
        # Generate and add synthetic conversations to the vector database
        synthetic_conversations = self.generate_synthetic_conversations(categories, rows_per_category)
        for conversation in synthetic_conversations:
            text = conversation['Person'] + " " + conversation['Therapist']
            embedding = vector_database.generate_embedding(text)
            vector_database.add_embeddings(np.array([embedding]), [conversation])

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
        if action.output_type == OutputType.DATAPIPE:
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

            if task.output_type == OutputType.METADATA:
                meta = Meta(
                    path=result,
                    type=result.split(".")[-1],
                    description="\n".join(task.outputs)
                    + "\n".join(task.outputs)
                    + "\n"
                    + "\n".join(task_inputs),
                    tag="task_output",
                )
                self.meta_data.append(meta)
                self.current_meta_data.append(meta)
            elif not task.executor_task:
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
                raise ReturnDirectException(result)
            return result  # , task.return_direct
        except ReturnDirectException as e:
            print('Inside return direct except\n')
            print('message: ', e.message)
            #raise ReturnDirectException(e.message)
            raise
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
                    action.output_type == OutputType.DATAPIPE
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
    
    def generate_follow_up_questions(self, follow_up_instructions: str, query: str) -> List[str]:
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

        # Retrieve the top matching conversations from the synthetic data knowledge base
        retrieved_conversations = self.vector_database.retrieve(query, top_k=3)

        # Extract retrieved therapist responses for context
        retrieved_context = ""
        for conversation in retrieved_conversations:
            retrieved_context += f"Category: {conversation['Category']}\n"
            retrieved_context += f"Person: {conversation['Person']}\n"
            retrieved_context += f"Therapist: {conversation['Therapist']}\n\n"

        # Step 2: Define a prompt template for generating follow-up questions
        prompt_template = (
            "You are an empathetic conversational agent tasked with generating appropriate follow-up questions "
            "for someone expressing a specific level of suicidal risk. Please generate three highly empathetic "
            "and supportive follow-up questions based on the instructions and context provided below. Ensure that the questions "
            "encourage dialogue without being intrusive.\n\n"
            "Instructions: {instructions}\n"
            "Context from Similar Conversations:\n{context}\n"
            "Follow-up questions:"
        )
        prompt = prompt_template.format(instructions=follow_up_instructions, context=retrieved_context)

        # # Few-shot examples for additional context (optional, can be used to further refine the prompt)
        # few_shot_examples = (
        #     "\n### Example 1: Suicidal Ideation (ID)\n"
        #     "Instructions: The user is expressing suicidal thoughts or ideation. Provide empathetic support and recommend professional help if appropriate.\n"
        #     "Follow-up questions:\n"
        #     "1. Could you share more about what you've been experiencing lately that's led you to feel this way?\n"
        #     "2. What are some things that usually help when you start feeling overwhelmed?\n"
        #     "3. Have you been able to talk to anyone about these thoughts? If not, would you consider it?\n"
        # )
        # prompt += few_shot_examples

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
            risk_assessment_result = self.suicidal_sensor.execute([query])
            print(f"Risk Assessment Result: {risk_assessment_result}")

            # Step 2: Determine methodology based on risk level
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
            follow_up_questions = self.generate_follow_up_questions(risk_methodology["follow_up_instructions"], query)
            print(f"Generated Follow-up Questions: {follow_up_questions}")

            # Step 4: Update history with follow-up questions
            #history += f"Follow-up Questions: {follow_up_questions}\n"

            print('history: ', history)
            print('end of history')

            return "\n".join([risk_methodology["response_start"], *follow_up_questions])

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
                vars = {}
                exec(actions, locals(), vars)
                final_response = (
                    self._prepare_planner_response_for_response_generator()
                )
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
