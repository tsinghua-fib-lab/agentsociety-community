"""
Lazy import like langchain-community.

How to add a new function:
1. Add a new file in the directory to define your function.
2. add a _import_xxx function to import the function.
3. add a __getattr__ function to lazy import the function.
4. add the citizen class to __all__ variable.
5. add the citizen class to the return value of get_type_to_cls_dict function.
"""

from typing import Callable, Dict, Type, TYPE_CHECKING

from agentsociety.simulation import AgentSociety

FunctionType = Callable[[AgentSociety], None]

if TYPE_CHECKING:
    from .donothing import do_nothing


def _import_do_nothing() -> Type[FunctionType]:
    from .donothing import do_nothing

    return do_nothing


def __getattr__(name: str) -> Type[FunctionType]:
    if name == "do_nothing":
        return _import_do_nothing()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["do_nothing"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[FunctionType]]]:
    """
    Use this function to get all the functions.
    """
    return {
        "do_nothing": _import_do_nothing,
    }
