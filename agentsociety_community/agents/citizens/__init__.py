"""
Lazy import like langchain-community.

How to add a new citizen:
1. Add a new file in the directory to define your citizen class.
2. add a _import_xxx function to import the citizen class.
3. add a __getattr__ function to lazy import the citizen class.
4. add the citizen class to __all__ variable.
5. add the citizen class to the return value of get_type_to_cls_dict function.
"""

from typing import Callable, Dict, Type, TYPE_CHECKING

from agentsociety.agent import CitizenAgentBase

if TYPE_CHECKING:
    from .example import ExampleCitizen


def _import_example_citizen() -> Type[CitizenAgentBase]:
    from .example import ExampleCitizen

    return ExampleCitizen


def __getattr__(name: str) -> Type[CitizenAgentBase]:
    if name == "ExampleCitizen":
        return _import_example_citizen()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["ExampleCitizen"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[CitizenAgentBase]]]:
    """
    Use this function to get all the citizen classes.
    """
    return {
        "ExampleCitizen": _import_example_citizen,
    }
