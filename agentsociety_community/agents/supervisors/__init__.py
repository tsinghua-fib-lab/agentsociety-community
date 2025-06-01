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
    from .bdsc2025_track_two_supervisor import (BaselineSupervisor,
                                                BDSC2025SupervisorBase)


def _import_baseline_supervisor() -> Type[SupervisorBase]:
    from .bdsc2025_track_two_supervisor import BaselineSupervisor

    return BaselineSupervisor


def _import_bdsc_2025_supervisor_base() -> Type[SupervisorBase]:
    from .bdsc2025_track_two_supervisor import BDSC2025SupervisorBase

    return BDSC2025SupervisorBase


def __getattr__(name: str) -> Type[SupervisorBase]:
    if name == "BaselineSupervisor":
        return _import_baseline_supervisor()
    if name == "BDSC2025SupervisorBase":
        return _import_bdsc_2025_supervisor_base()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["BaselineSupervisor", "BDSC2025SupervisorBase"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[SupervisorBase]]]:
    """
    Use this function to get all the supervisor classes.
    """
    return {
        "BaselineSupervisor": _import_baseline_supervisor,
        "BDSC2025SupervisorBase": _import_bdsc_2025_supervisor_base,
    }
