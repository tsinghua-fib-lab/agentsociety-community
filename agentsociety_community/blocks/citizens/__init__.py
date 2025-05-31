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
    from .cityagent import (
        MobilityBlock,
        SocialBlock,
        EconomyBlock,
        OtherBlock,
    )


def _import_cityagent_mobility_block() -> Type[Block]:
    from .cityagent import MobilityBlock

    return MobilityBlock


def _import_cityagent_social_block() -> Type[Block]:
    from .cityagent import SocialBlock

    return SocialBlock


def _import_cityagent_economy_block() -> Type[Block]:
    from .cityagent import EconomyBlock

    return EconomyBlock


def _import_cityagent_other_block() -> Type[Block]:
    from .cityagent import OtherBlock

    return OtherBlock


def __getattr__(name: str) -> Type[Block]:
    if name == "MobilityBlock":
        return _import_cityagent_mobility_block()
    elif name == "SocialBlock":
        return _import_cityagent_social_block()
    elif name == "EconomyBlock":
        return _import_cityagent_economy_block()
    elif name == "OtherBlock":
        return _import_cityagent_other_block()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = ["MobilityBlock", "SocialBlock", "EconomyBlock", "OtherBlock"]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[Block]]]:
    """
    Use this function to get all the citizen classes.
    """
    return {
        "MobilityBlock": _import_cityagent_mobility_block,
        "SocialBlock": _import_cityagent_social_block,
        "EconomyBlock": _import_cityagent_economy_block,
        "OtherBlock": _import_cityagent_other_block,
    }
