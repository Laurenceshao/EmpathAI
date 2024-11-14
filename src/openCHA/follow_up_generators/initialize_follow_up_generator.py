from typing import Any

from openCHA.llms import BaseLLM
from openCHA.llms import LLM_TO_CLASS
from openCHA.llms import LLMType
from openCHA.follow_up_generators import (
    BaseFollowUpGenerator,
)
from openCHA.follow_up_generators import FOLLOW_UP_GENERATOR_TO_CLASS
from openCHA.follow_up_generators import (
    FollowUpGeneratorType,
)


def initialize_follow_up_generator(
    llm: str = LLMType.OPENAI,
    follow_up_generator: str = FollowUpGeneratorType.BASE_GENERATOR,
    prefix: str = "",
    **kwargs: Any,
) -> BaseFollowUpGenerator:
    """
    This method provides a convenient way to initialize a response generator based on the specified language model type
    and response generator type. It handles the instantiation of the language model and the response generator class.

    Args:
        llm (str): Type of language model type to be used.
        response_generator (str): Type of response generator to be initialized.
        prefix (str): Prefix to be added to generated responses.
        **kwargs (Any): Additional keyword arguments.
    Return:
        BaseResponseGenerator: Initialized instance of the response generator.



    Example:
        .. code-block:: python

            from openCHA.llms import LLMType
            from openCHA.response_generators import ResponseGeneratorType
            response_generators = initialize_planner(llm=LLMType.OPENAI, response_generator=ResponseGeneratorType.BASE_GENERATOR)

    """

    if follow_up_generator not in FOLLOW_UP_GENERATOR_TO_CLASS:
        raise ValueError(
            f"Got unknown planner type: {follow_up_generator}. "
            f"Valid types are: {FOLLOW_UP_GENERATOR_TO_CLASS.keys()}."
        )

    if llm not in LLM_TO_CLASS:
        raise ValueError(
            f"Got unknown llm type: {llm}. "
            f"Valid types are: {LLM_TO_CLASS.keys()}."
        )

    follow_up_generator_cls = FOLLOW_UP_GENERATOR_TO_CLASS[
        follow_up_generator
    ]
    llm_model = LLM_TO_CLASS[llm]()
    follow_up_generator = follow_up_generator_cls(
        llm_model=llm_model, prefix=prefix
    )
    return follow_up_generator
