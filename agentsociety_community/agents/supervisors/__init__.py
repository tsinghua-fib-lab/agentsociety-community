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
from typing import Callable, Dict, Type, TYPE_CHECKING


if TYPE_CHECKING:
    from .governance.governance import GovernanceBase


def _import_governance_supervisor() -> Type[GovernanceBase]:
    from .governance.governance import GovernanceBase

    return GovernanceBase


def __getattr__(name: str) -> Type[GovernanceBase]:
    if name == "GovernanceBase":
        return _import_governance_supervisor()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["GovernanceBase"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[GovernanceBase]]]:
    """
    Use this function to get all the supervisor classes.
    """
    return {
        "GovernanceSupervisor": _import_governance_supervisor,
    } 