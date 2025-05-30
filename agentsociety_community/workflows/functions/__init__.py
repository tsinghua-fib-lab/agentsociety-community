"""
Lazy import like langchain-community.

How to add a new function:
1. Add a new file in the directory to define your function.
2. add a _import_xxx function to import the function.
3. add a __getattr__ function to lazy import the function.
4. add the citizen class to __all__ variable.
5. add the citizen class to the return value of get_type_to_cls_dict function.
"""

from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Type

from agentsociety.simulation import AgentSociety

FunctionType = Callable[[AgentSociety], Awaitable[None]]

if TYPE_CHECKING:
    from .bdsc_2025_track_two.workflows import (
        gather_survey_results_bdsc_2025_track_two,
        init_simulation_context_bdsc_2025_track_two,
        send_rumor_spread_survey_bdsc_2025_track_two)
    from .donothing import do_nothing


def _import_do_nothing() -> Type[FunctionType]:
    from .donothing import do_nothing

    return do_nothing


def _import_init_simulation_context_bdsc_2025_track_two() -> Type[FunctionType]:
    from .bdsc_2025_track_two.workflows import \
        init_simulation_context_bdsc_2025_track_two

    return init_simulation_context_bdsc_2025_track_two


def _import_send_rumor_spread_survey_bdsc_2025_track_two() -> Type[FunctionType]:
    from .bdsc_2025_track_two.workflows import \
        send_rumor_spread_survey_bdsc_2025_track_two

    return send_rumor_spread_survey_bdsc_2025_track_two


def _import_gather_survey_results_bdsc_2025_track_two() -> Type[FunctionType]:
    from .bdsc_2025_track_two.workflows import \
        gather_survey_results_bdsc_2025_track_two

    return gather_survey_results_bdsc_2025_track_two


def __getattr__(name: str) -> Type[FunctionType]:
    if name == "do_nothing":
        return _import_do_nothing()
    if name == "init_simulation_context_bdsc_2025_track_two":
        return _import_init_simulation_context_bdsc_2025_track_two()
    if name == "send_rumor_spread_survey_bdsc_2025_track_two":
        return _import_send_rumor_spread_survey_bdsc_2025_track_two()
    if name == "gather_survey_results_bdsc_2025_track_two":
        return _import_gather_survey_results_bdsc_2025_track_two()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "do_nothing",
    "init_simulation_context_bdsc_2025_track_two",
    "send_rumor_spread_survey_bdsc_2025_track_two",
    "gather_survey_results_bdsc_2025_track_two",
]


def get_type_to_cls_dict() -> Dict[str, Callable[[], Type[FunctionType]]]:
    """
    Use this function to get all the functions.
    """
    return {
        "do_nothing": _import_do_nothing,
        "init_simulation_context_bdsc_2025_track_two": _import_init_simulation_context_bdsc_2025_track_two,
        "send_rumor_spread_survey_bdsc_2025_track_two": _import_send_rumor_spread_survey_bdsc_2025_track_two,
        "gather_survey_results_bdsc_2025_track_two": _import_gather_survey_results_bdsc_2025_track_two,
    }
