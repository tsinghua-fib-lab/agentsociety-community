import json
from typing import Any

from agentsociety.configs.exp import (AgentFilterConfig, ExpConfig,
                                      WorkflowStepConfig, WorkflowType)
from agentsociety.environment import EnvironmentConfig
from agentsociety.simulation import AgentSociety

from ....agents.citizens.bdsc2025_track_two_envcitizen import \
    TrackTwoEnvCitizen
from .survey import create_rumor_spread_surveys, extract_survey_scores


async def init_simulation_context_bdsc_2025_track_two(simulation: AgentSociety):
    # initialize the simulationcontext
    simulation.context["survey_result"] = []
    simulation.context["final_score"] = {
        0: 0,
        "average": 0,
    }


async def send_and_gather_survey_results_bdsc_2025_track_two(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(TrackTwoEnvCitizen,))
    interceptor = simulation._message_interceptor
    assert interceptor is not None
    supervisor_name = interceptor.supervisor.__class__.__name__
    surveys = create_rumor_spread_surveys()
    simulation.context["final_score"]["average"] = 0
    for i, survey in enumerate(surveys):
        survey_responses_dict = await simulation.send_survey(survey, citizen_ids)
        # with open(f"{supervisor_name}_survey_responses_{i}.json", "w") as f:
        #     json.dump(survey_responses_dict, f, ensure_ascii=False)
        survey_scores, final_score = extract_survey_scores(
            list(survey_responses_dict.values())
        )
        simulation.context["survey_result"].append(
            {
                "survey_scores": survey_scores,
                "final_score": final_score,
                "rumor_index": i,
            }
        )
        simulation.context["final_score"][i] = final_score
        simulation.context["final_score"]["average"] += final_score
    simulation.context["final_score"]["average"] /= len(surveys)
    # with open(f"{supervisor_name}_simulation_context.json", "w") as f:
    #     json.dump(simulation.context, f, ensure_ascii=False)
    survey_request = await simulation.gather(
        target_agent_ids=citizen_ids, content="survey_request_history"
    )
    # with open(f"{supervisor_name}_survey_request.json", "w") as f:
    #     json.dump(survey_request, f, ensure_ascii=False)


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
            func=send_and_gather_survey_results_bdsc_2025_track_two,
            description="Send and gather the rumor spread survey.",
        ),
    ],
    environment=EnvironmentConfig(),
)
