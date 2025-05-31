import logging
import math
import random
from typing import Any, Optional

import jsonc
import numpy as np
from agentsociety.agent import Block, BlockContext, BlockParams, DotDict, FormatPrompt
from agentsociety.agent.dispatcher import BlockDispatcher
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from pydantic import Field

from ....agents.citizens.cityagent.societyagent import SocietyAgentBlockOutput
from .utils import clean_json_response

# Prompt templates for LLM interactions
PLACE_TYPE_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------
Your output must be a single selection from {poi_category} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "shopping"
}}
"""

PLACE_SECOND_TYPE_SELECTION_PROMPT = """
As an intelligent decision system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------

Your output must be a single selection from {poi_category} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "shopping"
}}
"""

PLACE_ANALYSIS_PROMPT = """
As an intelligent analysis system, please determine the type of place the user needs to visit based on their input requirement.
User Plan: {plan}
User requirement: {intention}
Other information: 
-------------------------
{other_info}
-------------------------

Your output must be a single selection from {place_list} without any additional text or explanation.

Please response in json format (Do not return any other text), example:
{{
    "place_type": "home"
}}
"""

RADIUS_PROMPT = """As an intelligent decision system, please determine the maximum travel radius (in meters) based on the current emotional state.

Current weather: ${context.weather}
Current temperature: ${context.temperature}
Your current emotion: ${context.current_emotion}
Your current thought: ${context.current_thought}
Other information: 
-------------------------
${context.other_information}
-------------------------

Please analyze how these emotions would affect travel willingness and return only a single integer number between 3000-200000 representing the maximum travel radius in meters. A more positive emotional state generally leads to greater willingness to travel further.

Please response in json format (Do not return any other text), example:
{{
    "radius": 10000
}}
"""


def gravity_model(pois):
    """
    Calculate selection probabilities for POIs using a gravity model.

    The model considers both distance decay (prefer closer locations)
    and spatial density (avoid overcrowded areas). Distances are grouped
    into 1km bins up to 10km, with POIs beyond 10km in a 'more' category.

    Args:
        pois: List of POI tuples containing (poi_data, distance)

    Returns:
        List of tuples: (name, id, normalized_weight, distance)
        with selection probabilities based on gravity model
    """
    # Initialize distance bins
    pois_Dis = {f"{d}k": [] for d in range(1, 11)}
    pois_Dis["more"] = []

    # Categorize POIs into distance bins
    for poi in pois:
        classified = False
        for d in range(1, 11):
            if (d - 1) * 1000 <= poi[1] < d * 1000:
                pois_Dis[f"{d}k"].append(poi)
                classified = True
                break
        if not classified:
            pois_Dis["more"].append(poi)

    res = []
    distanceProb = []
    # Calculate weights for each POI
    for poi in pois:
        for d in range(1, 11):
            if (d - 1) * 1000 <= poi[1] < d * 1000:
                n = len(pois_Dis[f"{d}k"])
                # Calculate ring area between (d-1)km and d km
                S = math.pi * ((d * 1000) ** 2 - ((d - 1) * 1000) ** 2)
                density = n / S  # POIs per square meter
                distance = max(poi[1], 1)  # Avoid division by zero

                # Inverse square distance decay combined with density
                weight = density / (distance**2)
                res.append((poi[0]["name"], poi[0]["id"], weight, distance))
                distanceProb.append(1 / math.sqrt(distance))
                break

    # Normalize probabilities and sample
    distanceProb = np.array(distanceProb)
    distanceProb /= distanceProb.sum()

    # Randomly sample 50 candidates weighted by distance probabilities
    sample_indices = np.random.choice(len(res), size=50, p=distanceProb)
    sampled_pois = [res[i] for i in sample_indices]

    # Normalize weights for final selection
    total_weight = sum(item[2] for item in sampled_pois)
    return [
        (item[0], item[1], item[2] / total_weight, item[3]) for item in sampled_pois
    ]


class PlaceSelectionBlock(Block):
    """
    Block for selecting destinations based on user intention.

    Implements a two-stage selection process:
    1. Select primary POI category (e.g., 'shopping')
    2. Select sub-category (e.g., 'bookstore')
    Uses LLM for decision making with fallback to random selection.

    Configurable Fields:
        search_limit: Max number of POIs to retrieve from map service
    """

    name = "PlaceSelectionBlock"
    description = "Selects destinations for unknown locations (excluding home/work)"

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        search_limit: int = 50,
    ):
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=agent_memory,
        )
        self.typeSelectionPrompt = FormatPrompt(PLACE_TYPE_SELECTION_PROMPT)
        self.secondTypeSelectionPrompt = FormatPrompt(
            PLACE_SECOND_TYPE_SELECTION_PROMPT
        )
        self.radiusPrompt = FormatPrompt(
            RADIUS_PROMPT,
            memory=agent_memory,
        )
        self.search_limit = search_limit  # Default config value

    async def forward(self, context: DotDict):
        """Execute the destination selection workflow"""
        # Stage 1: Select primary POI category
        poi_cate = self.environment.get_poi_cate()
        await self.typeSelectionPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            poi_category=list(poi_cate.keys()),
            other_info=self.environment.environment.get("other_information", "None"),
        )
        try:
            # LLM-based category selection
            levelOneType = await self.llm.atext_request(
                self.typeSelectionPrompt.to_dialog(),
                response_format={"type": "json_object"},
            )
            levelOneType = jsonc.loads(clean_json_response(levelOneType))["place_type"]
            sub_category = poi_cate[levelOneType]
        except Exception as e:
            get_logger().warning(f"Level 1 selection failed: {e}")
            levelOneType = random.choice(list(poi_cate.keys()))
            sub_category = poi_cate[levelOneType]

        # Stage 2: Select sub-category
        try:
            await self.secondTypeSelectionPrompt.format(
                plan=context["plan_context"]["plan"],
                intention=context["current_step"]["intention"],
                poi_category=sub_category,
                other_info=self.environment.environment.get(
                    "other_information", "None"
                ),
            )
            levelTwoType = await self.llm.atext_request(
                self.secondTypeSelectionPrompt.to_dialog(),
                response_format={"type": "json_object"},
            )
            levelTwoType = jsonc.loads(clean_json_response(levelTwoType))["place_type"]
        except Exception as e:
            get_logger().warning(f"Level 2 selection failed: {e}")
            levelTwoType = random.choice(sub_category)

        # Get travel radius from LLM
        try:
            await self.radiusPrompt.format(context=context)
            radius = await self.llm.atext_request(
                self.radiusPrompt.to_dialog(), response_format={"type": "json_object"}
            )
            radius = int(jsonc.loads(radius)["radius"])
        except Exception as e:
            get_logger().warning(f"Radius selection failed: {e}")
            radius = 10000  # Default 10km

        # Query and select POI
        xy = (await self.memory.status.get("position"))["xy_position"]
        center = (xy["x"], xy["y"])
        pois = self.environment.map.query_pois(
            center=center,
            category_prefix=levelTwoType,
            radius=radius,
            limit=self.search_limit,
        )

        if pois:
            pois = gravity_model(pois)
            probabilities = [item[2] for item in pois]
            selected = np.random.choice(len(pois), p=probabilities)
            next_place = (pois[selected][0], pois[selected][1])
        else:  # Fallback random selection
            all_pois = self.environment.map.get_all_pois()
            next_place = random.choice(all_pois)
            next_place = (next_place["name"], next_place["id"])

        context["next_place"] = next_place
        node_id = await self.memory.stream.add_mobility(
            description=f"For {context['current_step']['intention']}, selected: {next_place}"
        )
        return {
            "success": True,
            "evaluation": f"Selected destination: {next_place}",
            "consumed_time": 5,
            "node_id": node_id,
        }


class MoveBlock(Block):
    """Block for executing mobility operations (home/work/other)"""

    name = "MoveBlock"
    description = "Executes mobility operations between locations"

    def __init__(self, llm: LLM, environment: Environment, agent_memory: Memory):
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=agent_memory,
        )
        self.placeAnalysisPrompt = FormatPrompt(PLACE_ANALYSIS_PROMPT)

    async def forward(self, context: DotDict):
        agent_id = await self.memory.status.get("id")
        place_knowledge = await self.memory.status.get("location_knowledge")
        known_places = list(place_knowledge.keys())
        places = ["home", "workplace"] + known_places + ["other"]
        await self.placeAnalysisPrompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            place_list=places,
            other_info=self.environment.environment.get("other_information", "None"),
        )
        response = await self.llm.atext_request(
            self.placeAnalysisPrompt.to_dialog(),
            response_format={"type": "json_object"},
        )  #
        try:
            response = clean_json_response(response)
            response = jsonc.loads(response)["place_type"]
        except Exception as e:
            get_logger().warning(
                f"Place Analysis: wrong type of place, raw response: {response}"
            )
            response = "home"
        if response == "home":
            # go back home
            home = await self.memory.status.get("home")
            home = home["aoi_position"]["aoi_id"]
            nowPlace = await self.memory.status.get("position")
            node_id = await self.memory.stream.add_mobility(
                description=f"I returned home"
            )
            if (
                "aoi_position" in nowPlace
                and nowPlace["aoi_position"]["aoi_id"] == home
            ):
                return {
                    "success": True,
                    "evaluation": f"Successfully returned home (already at home)",
                    "to_place": home,
                    "consumed_time": 0,
                    "node_id": node_id,
                }
            await self.environment.set_aoi_schedules(
                person_id=agent_id,
                target_positions=home,
            )
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": f"Successfully returned home",
                "to_place": home,
                "consumed_time": 45,
                "node_id": node_id,
            }
        elif response == "workplace":
            # back to workplace
            work = await self.memory.status.get("work")
            work = work["aoi_position"]["aoi_id"]
            nowPlace = await self.memory.status.get("position")
            node_id = await self.memory.stream.add_mobility(
                description=f"I went to my workplace"
            )
            if (
                "aoi_position" in nowPlace
                and nowPlace["aoi_position"]["aoi_id"] == work
            ):
                return {
                    "success": True,
                    "evaluation": f"Successfully reached the workplace (already at the workplace)",
                    "to_place": work,
                    "consumed_time": 0,
                    "node_id": node_id,
                }
            await self.environment.set_aoi_schedules(
                person_id=agent_id,
                target_positions=work,
            )
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": f"Successfully reached the workplace",
                "to_place": work,
                "consumed_time": 45,
                "node_id": node_id,
            }
        elif response in known_places:
            the_place = place_knowledge[response]["id"]
            nowPlace = await self.memory.status.get("position")
            node_id = await self.memory.stream.add_mobility(
                description=f"I went to {response}"
            )
            if (
                "aoi_position" in nowPlace
                and nowPlace["aoi_position"]["aoi_id"] == the_place
            ):
                return {
                    "success": True,
                    "evaluation": f"Successfully reached {response} (already at {response})",
                    "to_place": the_place,
                    "consumed_time": 0,
                    "node_id": node_id,
                }
            await self.environment.set_aoi_schedules(
                person_id=agent_id,
                target_positions=the_place,
            )
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": f"Successfully reached {response}",
                "to_place": the_place,
                "consumed_time": 45,
                "node_id": node_id,
            }
        else:
            # move to other places
            next_place = context.get("next_place", None)
            nowPlace = await self.memory.status.get("position")
            node_id = await self.memory.stream.add_mobility(
                description=f"I went to {next_place}"
            )
            if next_place != None:
                await self.environment.set_aoi_schedules(
                    person_id=agent_id,
                    target_positions=next_place[1],
                )
            else:
                aois = self.environment.map.get_all_aois()
                while True:
                    r_aoi = random.choice(aois)
                    if len(r_aoi["poi_ids"]) > 0:
                        r_poi = random.choice(r_aoi["poi_ids"])
                        break
                poi = self.environment.map.get_poi(r_poi)
                next_place = (poi["name"], poi["aoi_id"])
                await self.environment.set_aoi_schedules(
                    person_id=agent_id,
                    target_positions=next_place[1],
                )
            number_poi_visited = await self.memory.status.get("number_poi_visited")
            number_poi_visited += 1
            await self.memory.status.update("number_poi_visited", number_poi_visited)
            return {
                "success": True,
                "evaluation": f"Successfully reached the destination: {next_place}",
                "to_place": next_place[1],
                "consumed_time": 45,
                "node_id": node_id,
            }


class MobilityNoneBlock(Block):
    """
    MobilityNoneBlock
    """

    name = "MobilityNoneBlock"
    description = "Handles other mobility operations"

    def __init__(self, llm: LLM, agent_memory: Memory):
        super().__init__(
            llm=llm,
            agent_memory=agent_memory,
        )

    async def forward(self, context: DotDict):
        """Log completion without action"""
        node_id = await self.memory.stream.add_mobility(
            description=f"I finished {context['current_step']['intention']}"
        )
        return {
            "success": True,
            "evaluation": f"Finished executing {context['current_step']['intention']}",
            "consumed_time": 0,
            "node_id": node_id,
        }


class MobilityBlockParams(BlockParams):
    # PlaceSelection
    radius_prompt: str = Field(
        default=RADIUS_PROMPT, description="Used to determine the maximum travel radius"
    )
    search_limit: int = Field(
        default=50, description="Number of POIs to retrieve from map service"
    )


class MobilityBlockContext(BlockContext):
    next_place: Optional[tuple[str, int]] = Field(
        default=None, description="The next place to go"
    )


class MobilityBlock(Block):
    """
    Main mobility coordination block.
    """

    ParamsType = MobilityBlockParams
    OutputType = SocietyAgentBlockOutput
    ContextType = MobilityBlockContext
    name = "MobilityBlock"
    description = "Responsible for all kinds of mobility-related operations"
    actions = {
        "place_selection": "Support the place selection action",
        "move": "Support the move action",
        "mobility_none": "Support other mobility operations",
    }

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        block_params: Optional[MobilityBlockParams] = None,
    ):
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        # initialize all blocks
        self.place_selection_block = PlaceSelectionBlock(
            llm, environment, agent_memory, self.params.search_limit
        )
        self.move_block = MoveBlock(llm, environment, agent_memory)
        self.mobility_none_block = MobilityNoneBlock(llm, agent_memory)
        self.trigger_time = 0  # Block invocation counter
        self.token_consumption = 0  # LLM token tracker

        # Initialize block routing system
        self.dispatcher = BlockDispatcher(llm, agent_memory)
        # register all blocks
        self.dispatcher.register_blocks(
            [self.place_selection_block, self.move_block, self.mobility_none_block]
        )

    async def forward(self, agent_context: DotDict) -> SocietyAgentBlockOutput:
        """Main entry point - delegates to sub-blocks"""
        self.trigger_time += 1
        context = agent_context | self.context
        # Select the appropriate sub-block using dispatcher
        selected_block = await self.dispatcher.dispatch(context)
        if selected_block is None:
            return self.OutputType(
                success=False,
                evaluation=f"Failed to {agent_context['current_step']['intention']}",
                consumed_time=random.randint(1, 30),
                node_id=None,
            )
        # Execute the selected sub-block and get the result
        result = await selected_block.forward(context)  #

        return self.OutputType(**result)
