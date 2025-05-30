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
    from .bdsc2025_track_two_supervisor.do_nothing_supervisor import \
        DoNothingSupervisor
    from .bdsc2025_track_two_supervisor.supervisor import BDSC2025Supervisor


def _import_bdsc_2025_supervisor() -> Type[SupervisorBase]:
    from .bdsc2025_track_two_supervisor.supervisor import BDSC2025Supervisor

    return BDSC2025Supervisor


def _import_do_nothing_supervisor() -> Type[SupervisorBase]:
    from .bdsc2025_track_two_supervisor.do_nothing_supervisor import \
        DoNothingSupervisor

    return DoNothingSupervisor


def __getattr__(name: str) -> Type[SupervisorBase]:
    if name == "BDSC2025Supervisor":
        return _import_bdsc_2025_supervisor()
    if name == "DoNothingSupervisor":
        return _import_do_nothing_supervisor()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["BDSC2025Supervisor", "DoNothingSupervisor"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[SupervisorBase]]]:
    """
    Use this function to get all the supervisor classes.
    """
    return {
        "BDSC2025Supervisor": _import_bdsc_2025_supervisor,
        "DoNothingSupervisor": _import_do_nothing_supervisor,
    }
