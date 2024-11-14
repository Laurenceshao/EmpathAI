from typing import Dict
from typing import Type

from openCHA.follow_up_generators import (
    BaseFollowUpGenerator,
)
from openCHA.follow_up_generators import (
    FollowUpGeneratorType,
)


FOLLOW_UP_GENERATOR_TO_CLASS: Dict[
    FollowUpGeneratorType, Type[BaseFollowUpGenerator]
] = {FollowUpGeneratorType.BASE_GENERATOR: BaseFollowUpGenerator}
