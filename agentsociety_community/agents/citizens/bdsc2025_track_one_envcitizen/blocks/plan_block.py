import logging
import random
from typing import Dict, List, Optional, Tuple

import jsonc

from agentsociety.agent import Block, FormatPrompt
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from .utils import clean_json_response

GUIDANCE_SELECTION_PROMPT = """As an intelligent agent's decision system, please help me determine a suitable option to satisfy my current need.
The Environment will influence the choice of steps.

Current weather: {weather}
Current temperature: {temperature}
Other information: 
-------------------------
{other_info}
-------------------------

Current need: Need to satisfy {current_need}
Current location: {current_location}
Current time: {current_time}
My income/consumption level: {consumption_level}
My occupation: {occupation}
My age: {age}
My emotion: {emotion_types}
My thought: {thought}

Guidance Options: 
-------------------------
{options}
-------------------------

Please evaluate and select the most appropriate option based on these three dimensions:
1. Attitude: Personal preference and evaluation of the option
2. Subjective Norm: Social environment and others' views on this behavior
3. Perceived Control: Difficulty and controllability of executing this option

Please response in json format (Do not return any other text), example:
{{
    "selected_option": "Select the most suitable option from Guidance Options and extent the option if necessary (or do things that can satisfy your needs or actions unless there is no specific options)",
    "evaluation": {{
        "attitude": "Attitude score for the option (0-1)",
        "subjective_norm": "Subjective norm score (0-1)", 
        "perceived_control": "Perceived control score (0-1)",
        "reasoning": "Specific reasons for selecting this option"
    }}
}}
"""

DETAILED_PLAN_PROMPT = """As an intelligent agent's plan system, please help me generate specific execution steps based on the selected guidance plan. 
The Environment will influence the choice of steps.

Current weather: {weather}
Current temperature: {temperature}
Other information: 
-------------------------
{other_info}
-------------------------

Selected plan: {selected_option}
Current location: {current_location} 
Current time: {current_time}
My income/consumption level: {consumption_level}
My occupation: {occupation}
My age: {age}
My emotion: {emotion_types}
My thought: {thought}

Notes:
1. type can only be one of these four: mobility, social, economy, other
    1.1 mobility: Decisions or behaviors related to large-scale spatial movement, such as location selection, going to a place, etc.
    1.2 social: Decisions or behaviors related to social interaction, such as finding contacts, chatting with friends, etc.
    1.3 economy: Decisions or behaviors related to shopping, work, etc.
    1.4 other: Other types of decisions or behaviors, such as small-scale activities, learning, resting, entertainment, etc.
2. steps should only include steps necessary to fulfill the target (limited to {max_plan_steps} steps)
3. intention in each step should be concise and clear
4. consumed_time is the time consumed for each step, in minutes

Please response in json format (Do not return any other text), example:
{{
    "plan": {{
        "target": "Eat at home",
        "steps": [
            {{
                "intention": "Return home from current location",
                "type": "mobility",
                "consumed_time": 10
            }},
            {{
                "intention": "Cook food",
                "type": "other",
                "consumed_time": 10
            }},
            {{
                "intention": "Have meal",
                "type": "other",
                "consumed_time": 10
            }}
        ]
    }}
}}

{{
    "plan": {{
        "target": "Eat outside",
        "steps": [
            {{
                "intention": "Select restaurant",
                "type": "mobility",
                "consumed_time": 10
            }},
            {{
                "intention": "Go to restaurant",
                "type": "mobility",
                "consumed_time": 10
            }},
            {{
                "intention": "Order food",
                "type": "economy",
                "consumed_time": 10
            }},
            {{
                "intention": "Have meal",
                "type": "other",
                "consumed_time": 10
            }}
        ]
    }}
}}
"""


class EnvPlanBlock(Block):
    """A block for generating and managing execution plans through LLM-guided decision making.

    Attributes:
        configurable_fields: List of configurable parameter names
        default_values: Default values for configurable parameters
        fields_description: Human-readable descriptions for configurable parameters
        guidance_options: Predefined options mapped to specific needs
        max_plan_steps: Maximum allowed steps in generated plans (configurable)
    """

    def __init__(
            self, 
            llm: LLM, 
            environment: Environment, 
            agent_memory: Memory, 
            max_plan_steps: int = 6,
            detailed_plan_prompt: str = DETAILED_PLAN_PROMPT,
        ):
        """Initialize PlanBlock with required components.

        Args:
            llm: Language Model interface for decision making
            environment: Environment for contextual data
            memory: Agent's memory storage for status tracking
        """
        super().__init__(llm=llm, environment=environment, agent_memory=agent_memory)
        self.guidance_prompt = FormatPrompt(template=GUIDANCE_SELECTION_PROMPT)
        self.detail_prompt = FormatPrompt(template=detailed_plan_prompt)
        self.trigger_time = 0
        self.token_consumption = 0
        self.guidance_options = {
            "hungry": ["Eat at home", "Eat outside", "Eat at current location"],
            "tired": ["Sleep at home"],
            "safe": ["Work", "Shopping"],
            "social": ["Contact with friends"],
            "whatever": ["leisure and entertainment", "other", "stay at home"],
        }

        # configurable fields
        self.max_plan_steps = max_plan_steps

    async def select_guidance(self, current_need: str) -> Optional[Tuple[dict, str]]:
        """Select optimal guidance option using Theory of Planned Behavior evaluation.

        Args:
            current_need: The agent's current need to fulfill

        Returns:
            Optional[tuple[dict, str]]: Selected option with TPB evaluation scores and reasoning. None if no guidance option is selected by bad response from LLM.
        """
        cognition = None
        position_now = await self.memory.status.get("position")
        home_location = await self.memory.status.get("home")
        work_location = await self.memory.status.get("work")
        location_knowledge = await self.memory.status.get("location_knowledge")
        known_locations = [item["id"] for item in location_knowledge.values()]
        id_to_name = {
            info["id"]: f"{name}({info['description']})"
            for name, info in location_knowledge.items()
        }
        current_location = "Outside"
        if (
            "aoi_position" in position_now
            and position_now["aoi_position"] == home_location["aoi_position"]
        ):
            current_location = "At home"
        elif (
            "aoi_position" in position_now
            and position_now["aoi_position"] == work_location["aoi_position"]
        ):
            current_location = "At workplace"
        elif (
            "aoi_position" in position_now
            and position_now["aoi_position"] in known_locations
        ):
            current_location = id_to_name[position_now["aoi_position"]]
        day, current_time = self.environment.get_datetime(format_time=True)
        options = self.guidance_options.get(current_need, [])
        if len(options) == 0:
            options = "Do things that can satisfy your needs or actions."
        await self.guidance_prompt.format(
            weather=self.environment.sense("weather"),
            temperature=self.environment.sense("temperature"),
            other_info=self.environment.environment.get("other_information", "None"),
            current_need=current_need,
            options=options,
            current_location=current_location,
            current_time=current_time,
            consumption_level=await self.memory.status.get("consumption"),
            occupation=await self.memory.status.get("occupation"),
            age=await self.memory.status.get("age"),
            emotion_types=await self.memory.status.get("emotion_types"),
            thought=await self.memory.status.get("thought"),
        )

        response = await self.llm.atext_request(
            self.guidance_prompt.to_dialog(), response_format={"type": "json_object"}
        )  #
        retry = 3
        while retry > 0:
            try:
                result = jsonc.loads(clean_json_response(response))  #
                if "selected_option" not in result or "evaluation" not in result:
                    raise ValueError("Invalid guidance selection format")
                if (
                    "attitude" not in result["evaluation"]
                    or "subjective_norm" not in result["evaluation"]
                    or "perceived_control" not in result["evaluation"]
                    or "reasoning" not in result["evaluation"]
                ):
                    raise ValueError(
                        "Evaluation must include attitude, subjective_norm, perceived_control, and reasoning"
                    )
                cognition = f"I choose to {result['selected_option']} because {result['evaluation']['reasoning']}"
                return result, cognition
            except Exception as e:
                get_logger().warning(
                    f"Error parsing guidance selection response: {str(e)}"
                )
                retry -= 1
        return None

    async def generate_detailed_plan(self, selected_option: str) -> Optional[dict]:
        """Generate executable steps for selected guidance option.

        Args:
            selected_option: The chosen guidance option from select_guidance()

        Returns:
            dict: Structured plan with target and typed execution steps. None if no plan is generated by bad response from LLM.
        """
        position_now = await self.memory.status.get("position")
        home_location = await self.memory.status.get("home")
        work_location = await self.memory.status.get("work")
        current_location = "Out"
        if (
            "aoi_position" in position_now
            and position_now["aoi_position"] == home_location["aoi_position"]
        ):
            current_location = "At home"
        elif (
            "aoi_position" in position_now
            and position_now["aoi_position"] == work_location["aoi_position"]
        ):
            current_location = "At workplace"
        day, current_time = self.environment.get_datetime(format_time=True)
        await self.detail_prompt.format(
            weather=self.environment.sense("weather"),
            temperature=self.environment.sense("temperature"),
            other_info=self.environment.environment.get("other_information", "None"),
            selected_option=selected_option,
            current_location=current_location,
            current_time=current_time,
            consumption_level=await self.memory.status.get("consumption"),
            occupation=await self.memory.status.get("occupation"),
            age=await self.memory.status.get("age"),
            emotion_types=await self.memory.status.get("emotion_types"),
            thought=await self.memory.status.get("thought"),
            max_plan_steps=self.max_plan_steps,
        )

        response = await self.llm.atext_request(self.detail_prompt.to_dialog())
        retry = 3
        while retry > 0:
            try:
                result = jsonc.loads(clean_json_response(response))  #
                if (
                    "plan" not in result
                    or "target" not in result["plan"]
                    or "steps" not in result["plan"]
                ):
                    raise ValueError("Invalid plan format")
                for step in result["plan"]["steps"]:
                    if "intention" not in step or "type" not in step:
                        raise ValueError("Each step must have an intention and a type")
                    if step["type"] not in ["mobility", "social", "economy", "other"]:
                        raise ValueError(f"Invalid step type: {step['type']}")
                return result
            except Exception as e:
                get_logger().warning(f"Error parsing detailed plan: {str(e)}")
                get_logger().debug(
                    f"Error parsing detailed plan: {str(e)} with response={response}"
                )
                retry -= 1
        return None

    async def forward(self):
        """Main workflow: Guidance selection -> Plan generation -> Memory update"""
        cognition = None
        # Step 1: Select guidance plan
        current_need = await self.memory.status.get("current_need")
        select_guidance = await self.select_guidance(current_need)
        if not select_guidance:
            return None
        guidance_result, cognition = select_guidance

        # Step 2: Generate detailed plan
        detailed_plan = await self.generate_detailed_plan(
            guidance_result["selected_option"]
        )

        if not detailed_plan:
            await self.memory.status.update("current_plan", None)
            return None

        # Step 3: Update plan and current step
        steps = detailed_plan["plan"]["steps"]
        for step in steps:
            step["evaluation"] = {"status": "pending", "details": ""}

        plan = {
            "target": detailed_plan["plan"]["target"],
            "steps": steps,
            "index": 0,
            "completed": False,
            "failed": False,
            "stream_nodes": [],
            "guidance": guidance_result,  # Save the evaluation result of the plan selection
        }
        formated_steps = "\n".join(
            [f"{i}. {step['intention']}" for i, step in enumerate(plan["steps"], 1)]
        )
        formated_plan = f"""
Overall Target: {plan['target']}
Execution Steps: \n{formated_steps}
        """
        _, plan["start_time"] = self.environment.get_datetime(format_time=True)
        await self.memory.status.update("current_plan", plan)
        await self.memory.status.update("execution_context", {"plan": formated_plan})
        await self.memory.stream.add_cognition(description=cognition)
        return cognition
