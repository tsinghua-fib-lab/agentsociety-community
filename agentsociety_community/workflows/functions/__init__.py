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


def _import_insert_citizen_information_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        insert_citizen_information
    
    return insert_citizen_information


def _import_gather_survey_results_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        gather_survey_results
    
    return gather_survey_results


def _import_gather_carbon_emission_results_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        gather_carbon_emission_results
    
    return gather_carbon_emission_results


def _import_gather_promotion_results_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        gather_promotion_results
    
    return gather_promotion_results


def _import_gather_communication_history_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        gather_communication_history
    
    return gather_communication_history


def _import_delete_ambassador_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        delete_ambassador
    
    return delete_ambassador


def _import_start_emission_log_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        start_emission_log
    
    return start_emission_log


def _import_send_carbon_awareness_survey_bdsc_2025_track_one() -> Type[FunctionType]:
    from .bdsc_2025_track_one.workflow import \
        send_canbon_awareness_survey
    
    return send_canbon_awareness_survey



def __getattr__(name: str) -> Type[FunctionType]:
    if name == "do_nothing":
        return _import_do_nothing()
    if name == "insert_citizen_information_bdsc_2025_track_one":
        return _import_insert_citizen_information_bdsc_2025_track_one()
    if name == "gather_survey_results_bdsc_2025_track_one":
        return _import_gather_survey_results_bdsc_2025_track_one()
    if name == "gather_carbon_emission_results_bdsc_2025_track_one":
        return _import_gather_carbon_emission_results_bdsc_2025_track_one()
    if name == "gather_promotion_results_bdsc_2025_track_one":
        return _import_gather_promotion_results_bdsc_2025_track_one()
    if name == "gather_communication_history_bdsc_2025_track_one":
        return _import_gather_communication_history_bdsc_2025_track_one()
    if name == "delete_ambassador_bdsc_2025_track_one":
        return _import_delete_ambassador_bdsc_2025_track_one()
    if name == "start_emission_log_bdsc_2025_track_one":
        return _import_start_emission_log_bdsc_2025_track_one()
    if name == "send_carbon_awareness_survey_bdsc_2025_track_one":
        return _import_send_carbon_awareness_survey_bdsc_2025_track_one()
    if name == "init_simulation_context_bdsc_2025_track_two":
        return _import_init_simulation_context_bdsc_2025_track_two()
    if name == "send_rumor_spread_survey_bdsc_2025_track_two":
        return _import_send_rumor_spread_survey_bdsc_2025_track_two()
    if name == "gather_survey_results_bdsc_2025_track_two":
        return _import_gather_survey_results_bdsc_2025_track_two()
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = [
    "do_nothing",
    "insert_citizen_information_bdsc_2025_track_one",
    "gather_survey_results_bdsc_2025_track_one",
    "gather_carbon_emission_results_bdsc_2025_track_one",
    "gather_promotion_results_bdsc_2025_track_one",
    "gather_communication_history_bdsc_2025_track_one",
    "delete_ambassador_bdsc_2025_track_one",
    "start_emission_log_bdsc_2025_track_one",
    "send_canbon_awareness_survey_bdsc_2025_track_one",
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
        "insert_citizen_information_bdsc_2025_track_one": _import_insert_citizen_information_bdsc_2025_track_one,
        "gather_survey_results_bdsc_2025_track_one": _import_gather_survey_results_bdsc_2025_track_one,
        "gather_carbon_emission_results_bdsc_2025_track_one": _import_gather_carbon_emission_results_bdsc_2025_track_one,
        "gather_promotion_results_bdsc_2025_track_one": _import_gather_promotion_results_bdsc_2025_track_one,
        "gather_communication_history_bdsc_2025_track_one": _import_gather_communication_history_bdsc_2025_track_one,
        "delete_ambassador_bdsc_2025_track_one": _import_delete_ambassador_bdsc_2025_track_one,
        "start_emission_log_bdsc_2025_track_one": _import_start_emission_log_bdsc_2025_track_one,
        "send_canbon_awareness_survey_bdsc_2025_track_one": _import_send_carbon_awareness_survey_bdsc_2025_track_one,
    }
