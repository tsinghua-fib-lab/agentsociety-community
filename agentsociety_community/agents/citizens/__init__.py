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
    from .bdsc2025_track_two_envcitizen.envcitizen import EnvCitizen
    from .bdsc2025_track_two_rumor_spreader.rumor_spreader import RumorSpreader


def _import_env_citizen() -> Type[CitizenAgentBase]:
    from .bdsc2025_track_two_envcitizen.envcitizen import EnvCitizen

    return EnvCitizen


def _import_rumor_spreader() -> Type[CitizenAgentBase]:
    from .bdsc2025_track_two_rumor_spreader.rumor_spreader import RumorSpreader

    return RumorSpreader


def __getattr__(name: str) -> Type[CitizenAgentBase]:
    if name == "EnvCitizen":
        return _import_env_citizen()
    if name == "RumorSpreader":
        return _import_rumor_spreader()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["EnvCitizen", "RumorSpreader"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[CitizenAgentBase]]]:
    """
    Use this function to get all the citizen classes.
    """
    return {
        "EnvCitizen": _import_env_citizen,
        "RumorSpreader": _import_rumor_spreader,
    }
