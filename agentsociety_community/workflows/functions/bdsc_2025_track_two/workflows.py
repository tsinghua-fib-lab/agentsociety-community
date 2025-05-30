from typing import Any

from agentsociety.configs.exp import (AgentFilterConfig, ExpConfig,
                                      WorkflowStepConfig, WorkflowType)
from agentsociety.environment import EnvironmentConfig
from agentsociety.simulation import AgentSociety

from ....agents.citizens.bdsc2025_track_two_envcitizen import EnvCitizen
from .survey import create_rumor_spread_survey, extract_survey_scores


async def init_simulation_context_bdsc_2025_track_two(simulation: AgentSociety):
    # initialize the simulationcontext
    simulation.context["survey_result"] = {"final_score": 10}
    simulation.context["final_score"] = 10


async def send_rumor_spread_survey_bdsc_2025_track_two(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(EnvCitizen,))
    await simulation.send_survey(create_rumor_spread_survey(), citizen_ids)


async def gather_survey_results_bdsc_2025_track_two(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(EnvCitizen,))
    survey_responses_dict = await simulation.gather(
        "survey_result", citizen_ids, flatten=True, keep_id=True
    )
    survey_responses: list[Any] = []
    for citizen_id, responses in survey_responses_dict.items():  # type: ignore
        survey_responses.append(responses)
    survey_scores, final_score = extract_survey_scores(survey_responses)
    simulation.context["survey_result"] = {
        "survey_scores": survey_scores,
        "final_score": final_score,
    }
    simulation.context["final_score"] = final_score


TRACK_TWO_EXPERIMENT = ExpConfig(
    name="BDSC_Track_Two",
    workflow=[
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=init_simulation_context_bdsc_2025_track_two,
            description="Initialize the simulation context.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.STEP,
            steps=20,
            ticks_per_step=300,
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=send_rumor_spread_survey_bdsc_2025_track_two,
            description="Send the rumor spread survey.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=gather_survey_results_bdsc_2025_track_two,
            description="Gather the survey results.",
        ),
    ],
    environment=EnvironmentConfig(),
)
