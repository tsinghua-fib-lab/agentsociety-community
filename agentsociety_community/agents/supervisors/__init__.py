"""
Lazy import like langchain-community.

How to add a new supervisor:
1. Add a new file in the directory to define your supervisor class.
2. add a _import_xxx function to import the supervisor class.
3. add a __getattr__ function to lazy import the supervisor class.
4. add the supervisor class to __all__ variable.
5. add the supervisor class to the return value of get_type_to_cls_dict function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Type

from agentsociety.agent import SupervisorBase

if TYPE_CHECKING:
    from .bdsc2025_track_two_supervisor.supervisor import Supervisor


def _import_bdsc_2025_supervisor() -> Type[SupervisorBase]:
    from .bdsc2025_track_two_supervisor.supervisor import Supervisor

    return Supervisor


def __getattr__(name: str) -> Type[SupervisorBase]:
    if name == "Supervisor":
        return _import_bdsc_2025_supervisor()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["Supervisor"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[SupervisorBase]]]:
    """
    Use this function to get all the supervisor classes.
    """
    return {
        "Supervisor": _import_bdsc_2025_supervisor,
    }
