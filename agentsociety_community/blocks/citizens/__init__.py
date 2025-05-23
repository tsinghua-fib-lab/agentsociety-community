"""
Lazy import like langchain-community.

How to add a new block:
1. Add a new file in the directory to define your block class.
2. add a _import_xxx function to import the block class.
3. add a __getattr__ function to lazy import the block class.
4. add the block class to __all__ variable.
5. add the block class to the return value of get_type_to_cls_dict function.
"""

from typing import Callable, Dict, Type, TYPE_CHECKING

from agentsociety.agent import Block

if TYPE_CHECKING:
    from .example import ExampleCitizenBlock


def _import_example_citizen_block() -> Type[Block]:
    from .example import ExampleCitizenBlock

    return ExampleCitizenBlock


def __getattr__(name: str) -> Type[Block]:
    if name == "ExampleCitizenBlock":
        return _import_example_citizen_block()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["ExampleCitizenBlock"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[Block]]]:
    """
    Use this function to get all the citizen classes.
    """
    return {
        "ExampleCitizenBlock": _import_example_citizen_block,
    }
