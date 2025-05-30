"""
The challenge runner.
Input: Configs, Output: Evaluation results.
"""

from typing import Any
from agentsociety.simulation import AgentSociety
from agentsociety.configs.exp import WorkflowType, WorkflowStepConfig, AgentFilterConfig, ExpConfig
from agentsociety.environment import EnvironmentConfig

from ....agents.citizens.bdsc2025_track_one_envambassador import EnvAgentBase
from ....agents.citizens.bdsc2025_track_one_envcitizen import TrackOneEnvCitizen
from .survey import create_low_carbon_awareness_survey, extract_survey_scores


async def insert_citizen_information(simulation: AgentSociety):
    """
    Insert the citizen information into the ambassador's memory.
    """
    citizen_ids = await simulation.filter(types=(TrackOneEnvCitizen,))
    ambassador = (await simulation.filter(types=(EnvAgentBase,)))[0]
    await simulation.update([ambassador], "citizen_ids", citizen_ids)
    # gather profiles
    names = await simulation.gather("name", citizen_ids, flatten=True, keep_id=True)
    genders = await simulation.gather("gender", citizen_ids, flatten=True, keep_id=True)
    ages = await simulation.gather("age", citizen_ids, flatten=True, keep_id=True)
    educations = await simulation.gather("education", citizen_ids, flatten=True, keep_id=True)
    occupations = await simulation.gather("occupation", citizen_ids, flatten=True, keep_id=True)
    marriage_statuses = await simulation.gather("marriage_status", citizen_ids, flatten=True, keep_id=True)
    homes = await simulation.gather("home", citizen_ids, flatten=True, keep_id=True)
    workplaces = await simulation.gather("work", citizen_ids, flatten=True, keep_id=True)
    background_stories = await simulation.gather("background_story", citizen_ids, flatten=True, keep_id=True)
    citizen_profiles = {}
    for citizen_id in names.keys(): # type: ignore
        citizen_profiles[citizen_id] = {
            "name": names[citizen_id],
            "gender": genders[citizen_id],
            "age": ages[citizen_id],
            "education": educations[citizen_id],
            "occupation": occupations[citizen_id],
            "marriage_status": marriage_statuses[citizen_id],
            "home": homes[citizen_id]['aoi_position'],
            "workplace": workplaces[citizen_id]['aoi_position'],
            "background_story": background_stories[citizen_id]
        }

    # initialize the simulationcontext
    simulation.context["survey_result"] = {"final_score": 50}
    simulation.context["carbon_emission_result"] = {"final_score": 50}
    simulation.context["promotion_result"] = {"final_score": 1}
    simulation.context["communication_history"] = {}
    simulation.context["overall_score"] = 50
    await simulation.update([ambassador], "citizens", citizen_profiles)


async def gather_survey_results(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(TrackOneEnvCitizen,))
    survey_responses_dict = await simulation.gather(
        "survey_result", citizen_ids, flatten=True, keep_id=True
    )
    survey_responses: list[Any] = []
    for citizen_id, responses in survey_responses_dict.items():  # type: ignore
        survey_responses.append(responses)
    survey_scores, final_score = extract_survey_scores(survey_responses)
    simulation.context["survey_result"] = {
        "survey_scores": survey_scores,
        "final_score": final_score
    }


async def gather_carbon_emission_results(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(TrackOneEnvCitizen,))
    transportation_logs = await simulation.gather(
        "transportation_log", citizen_ids, flatten=True, keep_id=True
    )
    car_factor = 0.040
    public_transport_factor = 0.016
    walk_logs = []
    public_transport_logs = []
    car_logs = []
    for _, logs in transportation_logs.items():
        for log in logs:
            if log["mode"] == "walk":
                walk_logs.append(log)
            elif log["mode"] == "public transport":
                public_transport_logs.append(log)
            elif log["mode"] == "car":
                car_logs.append(log)
    # distance in km
    total_distance_car = sum([log["distance"] for log in car_logs])/1000
    total_distance_public_transport = sum([log["distance"] for log in public_transport_logs])/1000
    total_distance_walk = sum([log["distance"] for log in walk_logs])/1000
    total_distance = total_distance_car + total_distance_public_transport + total_distance_walk

    # carbon emission in kg
    carbon_emission_car = total_distance_car * car_factor
    carbon_emission_public_transport = total_distance_public_transport * public_transport_factor
    total_carbon_emission_baseline = total_distance * car_factor
    total_carbon_emission = carbon_emission_car + carbon_emission_public_transport
    final_score = (100 * (total_carbon_emission_baseline - total_carbon_emission) / total_carbon_emission_baseline) if total_carbon_emission_baseline > 0 else 0
    simulation.context["carbon_emission_result"] = {
        "walk_logs": walk_logs,
        "public_transport_logs": public_transport_logs,
        "car_logs": car_logs,
        "total_distance": total_distance,
        "total_carbon_emission": total_carbon_emission,
        "final_score": final_score
    }
    overall_score = (simulation.context["survey_result"]["final_score"]/2 + simulation.context["carbon_emission_result"]["final_score"]/2) * simulation.context["promotion_result"]["final_score"]
    simulation.context["overall_score"] = overall_score


async def gather_promotion_results(simulation: AgentSociety):
    ambassador = (await simulation.filter(types=(EnvAgentBase,)))[0]
    promotion_log = await simulation.gather(
        "probe_logs", [ambassador], flatten=True, keep_id=True
    )
    promotion_log = promotion_log[ambassador]
    message_logs = promotion_log["message"]
    poster_logs = promotion_log["poster"]
    announcement_logs = promotion_log["announcement"]
    message_score = 0
    for log in message_logs:
        message_score += (log["credibility"] + log["reasonableness"]) / 2
    message_score = (message_score / len(message_logs)) if len(message_logs) > 0 else 100
    poster_score = 0
    for log in poster_logs:
        poster_score += (log["credibility"] + log["reasonableness"]) / 2
    poster_score = (poster_score / len(poster_logs)) if len(poster_logs) > 0 else 100
    announcement_score = 0
    for log in announcement_logs:
        announcement_score += (log["credibility"] + log["reasonableness"]) / 2
    announcement_score = (announcement_score / len(announcement_logs)) if len(announcement_logs) > 0 else 100
    final_score = (0.1*message_score + 0.3*poster_score + 0.6*announcement_score)/100
    simulation.context["promotion_result"] = {
        "promotion_log": promotion_log,
        "message_score": message_score,
        "poster_score": poster_score,
        "announcement_score": announcement_score,
        "final_score": final_score
    }


async def gather_communication_history(simulation: AgentSociety):
    ambassador = (await simulation.filter(types=(EnvAgentBase,)))[0]
    communication_history = await simulation.gather(
        "chat_histories", [ambassador], flatten=True, keep_id=True
    )
    simulation.context["communication_history"] = communication_history[ambassador]


async def delete_ambassador(simulation: AgentSociety):
    ambassador = (await simulation.filter(types=(EnvAgentBase,)))[0]
    await simulation.delete_agents([ambassador])


async def start_emission_log(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(TrackOneEnvCitizen,))
    for citizen_id in citizen_ids:
        await simulation.update([citizen_id], "logging_flag", True)


async def send_canbon_awareness_survey(simulation: AgentSociety):
    citizen_ids = await simulation.filter(types=(TrackOneEnvCitizen,))
    await simulation.send_survey(create_low_carbon_awareness_survey(), citizen_ids)


TRACK_ONE_EXPERIMENT=ExpConfig(
    name="BDSC_Track_One",
    workflow=[
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=insert_citizen_information,
            description="Insert the citizen information into the ambassador's memory.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.RUN,
            days=1,
            ticks_per_step=1800,
            description="Run the simulation."
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=gather_promotion_results,
            description="Gather the promotion results.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=gather_communication_history,
            description="Gather the communication history.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=delete_ambassador,
            description="Delete the ambassador."
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=start_emission_log,
            description="Start the emission log."
        ),
        WorkflowStepConfig(
            type=WorkflowType.RUN,
            days=1,
            ticks_per_step=900,
            description="Run the simulation."
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=send_canbon_awareness_survey,
            description="Send the carbon awareness survey.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=gather_survey_results,
            description="Gather the survey results.",
        ),
        WorkflowStepConfig(
            type=WorkflowType.FUNCTION,
            func=gather_carbon_emission_results,
            description="Gather the carbon emission results.",
        ),
    ],
    environment=EnvironmentConfig(
        start_tick=8 * 60 * 60,
    ),
)