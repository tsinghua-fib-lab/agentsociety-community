from collections import deque
import random
import time
import os
import json

import jsonc
from typing import Optional

from pydantic import Field
from agentsociety.agent import AgentToolbox, Block, CitizenAgentBase, AgentParams, FormatPrompt, AgentContext, DotDict, StatusAttribute
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from agentsociety.survey import Survey
from .blocks import (EnvCognitionBlock, EnvNeedsBlock, EnvPlanBlock, EnvMobilityBlock, EnvEconomyBlock, EnvSocialBlock, EnvOtherBlock)
from .blocks.needs_block import INITIAL_NEEDS_PROMPT
from .blocks.plan_block import DETAILED_PLAN_PROMPT

def extract_json(output_str):
    """Extract JSON substring from a raw string response.

    Args:
        output_str: Raw string output that may contain JSON data.

    Returns:
        Extracted JSON string if valid, otherwise None.

    Note:
        Searches for the first '{' and last '}' to isolate JSON content.
        Catches JSON decoding errors and logs warnings.
    """
    try:
        # Find the positions of the first '{' and the last '}'
        start = output_str.find("{")
        end = output_str.rfind("}")

        # Extract the substring containing the JSON
        json_str = output_str[start : end + 1]

        # Convert the JSON string to a dictionary
        return json_str
    except (ValueError, jsonc.JSONDecodeError) as e:
        get_logger().warning(f"Failed to extract JSON: {e}")
        return None

ENVIRONMENT_REFLECTION_PROMPT = """
You are a citizen of the city.
Your background story: {background_story}
Your attitude towards the environmental protection: {environmental_attitude}

In your current location, you can sense the following information:
{get_aoi_info}

What's your feeling about those environmental information? 
"""


class EnvSocialAgentConfig(AgentParams):
    """Configuration for social agent."""
    enable_cognition: bool = Field(default=True, description="Whether to enable cognition")
    UBI: float = Field(default=0, description="Universal Basic Income")
    num_labor_hours: int = Field(default=168, description="Number of labor hours per month")
    productivity_per_labor: float = Field(default=1, description="Productivity per labor hour")
    time_diff: int = Field(default=30 * 24 * 60 * 60, description="Time difference between two triggers")

    # Maxlow's Needs
    need_initialization_prompt: str = Field(default=INITIAL_NEEDS_PROMPT, description="Initial needs prompt")

    # Planned-Behavior
    max_plan_steps: int = Field(default=6, description="Maximum number of steps in a plan")
    plan_generation_prompt: str = Field(default=DETAILED_PLAN_PROMPT, description="Plan generation prompt")


class TrackOneEnvCitizen(CitizenAgentBase):
    ParamsType = EnvSocialAgentConfig
    Context = AgentContext
    StatusAttributes = [
        StatusAttribute(name="type",type=str,default="citizen",description="agent's type"),
        StatusAttribute(name="environmental_attitude",type=str,default="",description="agent's attitude towards the environmental protection"),
        StatusAttribute(name="survey_result",type=str,default="",description="agent's survey result"),
        # Needs Model
        StatusAttribute(name="hunger_satisfaction",type=float,default=0.9,description="agent's hunger satisfaction, 0-1"),
        StatusAttribute(name="energy_satisfaction",type=float,default=0.9,description="agent's energy satisfaction, 0-1"),
        StatusAttribute(name="safety_satisfaction",type=float,default=0.4,description="agent's safety satisfaction, 0-1"),
        StatusAttribute(name="social_satisfaction",type=float,default=0.6,description="agent's social satisfaction, 0-1"),
        StatusAttribute(name="current_need",type=str,default="none",description="agent's current need"),
        # Plan Behavior Model
        StatusAttribute(name="current_plan",type=dict,default={},description="agent's current plan"),
        StatusAttribute(name="execution_context",type=dict,default={},description="agent's execution context"),
        StatusAttribute(name="plan_history",type=list,default=[],description="agent's plan history"),
        # cognition
        StatusAttribute(name="emotion",type=dict,default={"sadness": 5, "joy": 5, "fear": 5, "disgust": 5, "anger": 5, "surprise": 5},description="agent's emotion, 0-10"),
        StatusAttribute(name="attitude",type=dict,default={},description="agent's attitude"),
        StatusAttribute(name="thought",type=str,default="Currently nothing good or bad is happening",description="agent's thought",whether_embedding=True),
        StatusAttribute(name="emotion_types",type=str,default="Relief",description="agent's emotion types",whether_embedding=True),
        # economy
        StatusAttribute(name="work_skill",type=float,default=0.5,description="agent's work skill, 0-1"),
        StatusAttribute(name="tax_paid",type=float,default=0.0,description="agent's tax paid"),
        StatusAttribute(name="consumption_currency",type=float,default=0.0,description="agent's consumption currency"),
        StatusAttribute(name="goods_demand",type=int,default=0,description="agent's goods demand"),
        StatusAttribute(name="goods_consumption",type=int,default=0,description="agent's goods consumption"),
        StatusAttribute(name="work_propensity",type=float,default=0.0,description="agent's work propensity, 0-1"),
        StatusAttribute(name="consumption_propensity",type=float,default=0.0,description="agent's consumption propensity, 0-1"),
        StatusAttribute(name="to_consumption_currency",type=float,default=0.0,description="agent's to consumption currency"),
        # other
        StatusAttribute(name="firm_id",type=int,default=0,description="agent's firm id"),
        StatusAttribute(name="government_id",type=int,default=0,description="agent's government id"),
        StatusAttribute(name="bank_id",type=int,default=0,description="agent's bank id"),
        StatusAttribute(name="nbs_id",type=int,default=0,description="agent's nbs id"),
        StatusAttribute(name="firm_forward",type=int,default=0,description="agent's firm forward"),
        StatusAttribute(name="bank_forward",type=int,default=0,description="agent's bank forward"),
        StatusAttribute(name="nbs_forward",type=int,default=0,description="agent's nbs forward"),
        StatusAttribute(name="government_forward",type=int,default=0,description="agent's government forward"),
        StatusAttribute(name="forward",type=int,default=0,description="agent's forward"),
        StatusAttribute(name="depression",type=float,default=0.0,description="agent's depression, 0-1"),
        StatusAttribute(name="ubi_opinion",type=list,default=[],description="agent's ubi opinion"),
        StatusAttribute(name="working_experience",type=list,default=[],description="agent's working experience"),
        StatusAttribute(name="work_hour_month",type=float,default=160,description="agent's work hour per month"),
        StatusAttribute(name="work_hour_finish",type=float,default=0,description="agent's work hour finished"),
        # social
        StatusAttribute(name="friends",type=list,default=[],description="agent's friends list"),
        StatusAttribute(name="relationships",type=dict,default={},description="agent's relationship strength with each friend"),
        StatusAttribute(name="relation_types",type=dict,default={},description="agent's relation types with each friend"),
        StatusAttribute(name="chat_histories",type=dict,default={},description="all chat histories"),
        StatusAttribute(name="interactions",type=dict,default={},description="all interaction records"),
        # mobility
        StatusAttribute(name="number_poi_visited",type=int,default=1,description="agent's number of poi visited"),
        StatusAttribute(name="transportation_log",type=list,default=[],description="agent's transportation log"),
        StatusAttribute(name="logging_flag",type=bool,default=False,description="agent's logging flag"),
        StatusAttribute(name="location_knowledge",type=dict,default={},description="agent's location knowledge"),
    ]
    description: str = """The citizen agent used in the BDSC2025 competition - Track One"""

    """Agent implementation with configurable cognitive/behavioral modules and social interaction capabilities."""
    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[EnvSocialAgentConfig] = None,
        blocks: Optional[list[Block]] = None,
    ) -> None:
        """Initialize agent with core components and configuration."""
        super().__init__(
            id=id,
            name=name,
            toolbox=toolbox,
            memory=memory,
            agent_params=agent_params,
            blocks=blocks,
        )

        self.needs_block = EnvNeedsBlock(
            llm=self.llm, 
            environment=self.environment, 
            agent_memory=self.memory,
            initial_prompt=self.params.need_initialization_prompt,
        )

        self.plan_block = EnvPlanBlock(
            llm=self.llm, 
            environment=self.environment, 
            agent_memory=self.memory,
            max_plan_steps=self.params.max_plan_steps,
            detailed_plan_prompt=self.params.plan_generation_prompt,
        )

        self.cognition_block = EnvCognitionBlock(
            llm=self.llm, agent_memory=self.memory, environment=self.environment
        )
        self.environment_reflection_prompt = FormatPrompt(ENVIRONMENT_REFLECTION_PROMPT)
        self.step_count = -1
        self.cognition_update = -1
        self.last_attitude_update = None
        self.info_checked = {}

        env_mobility_block = EnvMobilityBlock(
            llm=self.llm,
            environment=self.environment,
            agent_memory=self.memory,
        )
        env_economy_block = EnvEconomyBlock(
            llm=self.llm,
            environment=self.environment,
            agent_memory=self.memory,
        )
        env_social_block = EnvSocialBlock(
            llm=self.llm,
            environment=self.environment,
            agent_memory=self.memory,
        )
        env_social_block.set_agent(self)
        env_other_blocks = EnvOtherBlock(  
            llm=self.llm,
            environment=self.environment,
            agent_memory=self.memory,
        )
        self.blocks = [env_mobility_block, env_economy_block, env_social_block, env_other_blocks]
        self.dispatcher.register_blocks(self.blocks)

    async def reflect_to_environment(self):
        """Reflect to the environment"""
        aoi_info = await self.get_aoi_info()
        if aoi_info and aoi_info != "":
            position = await self.memory.status.get("position")
            aoi_id = position["aoi_position"]["aoi_id"]
            if aoi_id not in self.info_checked or self.info_checked[aoi_id] == 0:
                await self.environment_reflection_prompt.format(self)
                reflection = await self.llm.atext_request(
                    self.environment_reflection_prompt.to_dialog(),
                )
                self.info_checked[aoi_id] = 10
                await self.save_agent_thought(reflection)
                await self.memory.stream.add_event(f"You seeing some posters about environmental protection, and you feel {reflection}")
            else:
                self.info_checked[aoi_id] -= 1

    async def generate_user_survey_response(self, survey: Survey) -> str:
        survey_prompt = survey.to_prompt()
        dialog = []

        # Add system prompt
        system_prompt = "Please answer the survey question in first person. Follow the format requirements strictly and provide clear and specific answers (In JSON format). For example: {{'Q1': 'A', 'Q2': 'B', 'Q3': 'C', 'Q4': 'D'}}"
        dialog.append({"role": "system", "content": system_prompt})

        # Add memory context
        if self.memory:
            attitude = await self.memory.status.get("environmental_attitude")
            dialog.append(
                {
                    "role": "system",
                    "content": f"Answer based on the following attitude towards the environmental protection and low carbon lifestyle:\n{attitude}",
                }
            )

        # Add survey question
        dialog.append({"role": "user", "content": survey_prompt})

        for retry in range(3):
            try:
                # Use LLM to generate a response
                # print(f"dialog: {dialog}")
                _response = await self.llm.atext_request(
                    dialog, response_format={"type": "json_object"}
                )
                json_str = extract_json(_response)
                if json_str:
                    json_dict = jsonc.loads(json_str)
                    json_str = jsonc.dumps(json_dict, ensure_ascii=False)
                    break
            except:
                pass
        else:
            import traceback

            traceback.print_exc()
            raise Exception("Failed to generate survey response")
        await self.memory.status.update("survey_result", json_str)
        return json_str

    async def reset(self):
        """Reset the agent."""
        # reset position to home
        await self.reset_position()

        # reset needs
        await self.memory.status.update("current_need", "none")

        # reset plans and actions
        await self.memory.status.update("current_plan", {})
        await self.memory.status.update("execution_context", {})

        # reset initial flag
        await self.needs_block.reset()

    async def update_environmental_attitude(self):
        """Update the environmental attitude"""
        try:
            if self.last_attitude_update is None:
                # initialize the attitude
                prompt = f"""
You are a citizen of the city.
Your background story: {await self.memory.status.get("background_story")}
What's your attitude towards the environmental protection? (Clear and concise, within 20 words)
"""
                attitude = await self.llm.atext_request(
                    [{"role": "user", "content": prompt}],
                )
                await self.memory.status.update("environmental_attitude", attitude)
                day, t = self.environment.get_datetime()
                self.last_attitude_update = (day, t)
            else:
                last_day, last_t = self.last_attitude_update
                current_day, current_t = self.environment.get_datetime()
                # Calculate total seconds since start of day (1 day = 86400 seconds)
                last_total_seconds = last_day * 86400 + last_t
                current_total_seconds = current_day * 86400 + current_t
                # Update environmental attitude if one hour (3600 seconds) has passed since last update
                if current_total_seconds - last_total_seconds >= 3600:
                    related_memories = await self.memory.stream.search(
                        query="environmental protection", top_k=10
                    )
                    prompt = f"""
You are a citizen of the city.
Your background story: {await self.memory.status.get("background_story")}
Your attitude towards the environmental protection: {await self.memory.status.get("environmental_attitude")}
Here are some related memories: \n{related_memories}
Please update your attitude towards the environmental protection based on the memories. (Clear and concise, within 20 words)
"""
                    attitude = await self.llm.atext_request(
                        [{"role": "user", "content": prompt}],
                    )
                    await self.memory.status.update("environmental_attitude", attitude)
                    self.last_attitude_update = (current_day, current_t)
        except Exception as e:
            pass

    async def plan_generation(self):
        """Generate a new plan if no current plan exists in memory."""
        cognition = None
        current_plan = await self.memory.status.get("current_plan")
        if current_plan is None or not current_plan:
            cognition = (
                await self.plan_block.forward()
            )  # Delegate to PlanBlock for plan creation
        return cognition

    # Main workflow
    async def forward(self):
        """Main agent loop coordinating status updates, plan execution, and cognition."""
        start_time = time.time()
        self.step_count += 1
        await self.update_environmental_attitude()

        # reflect to environment
        await self.reflect_to_environment()

        logging_flag = await self.memory.status.get("logging_flag")
        if not logging_flag:
            return

        # check last step
        ifpass = await self.check_and_update_step()
        if not ifpass:
            return
        get_logger().debug(
            f"Agent {self.id}: Finished main workflow - check and update step"
        )

        # Maxlow's Needs
        cognition = await self.needs_block.forward()
        if self.params.enable_cognition and cognition:
            await self.save_agent_thought(cognition)

        # Planned-Behavior
        cognition = await self.plan_generation()
        if cognition:
            await self.save_agent_thought(cognition)
        get_logger().debug(f"Agent {self.id}: Finished main workflow - plan")

        # step execution - dispatch to different blocks
        await self.step_execution()
        get_logger().debug(f"Agent {self.id}: Finished main workflow - step execution")

        # cognition
        if self.params.enable_cognition:
            await self.cognition_block.forward()
        get_logger().debug(f"Agent {self.id}: Finished main workflow - cognition")

        return time.time() - start_time

    async def check_and_update_step(self):
        """Check if the previous step has been completed"""
        status = await self.memory.status.get("status")
        if status == 2:
            # Agent is moving
            return False

        # Get the previous step information
        current_plan = await self.memory.status.get("current_plan")
        # If there is no current plan, return True
        if current_plan is None or not current_plan:
            return True
        step_index = current_plan.get("index", 0)
        current_step = current_plan.get("steps", [])[step_index]
        time_now = self.environment.get_tick()
        step_start_time = current_step["start_time"]
        step_consumed_time = current_step["evaluation"]["consumed_time"]
        try:
            time_end_plan = step_start_time + int(step_consumed_time) * 60
        except Exception as e:
            time_end_plan = time_now
        if time_now >= time_end_plan:
            # The previous step has been completed
            current_step["evaluation"]["consumed_time"] = (
                time_now - step_start_time
            ) / 60
            current_plan["stream_nodes"].append(current_step["evaluation"]["node_id"])
            if current_step["evaluation"]["success"]:
                # last step is completed
                current_plan["steps"][step_index] = current_step
                if step_index + 1 < len(current_plan["steps"]):
                    # Last step is completed
                    current_plan["index"] = step_index + 1
                    await self.memory.status.update("current_plan", current_plan)
                else:
                    # Whole plan is completed
                    current_plan["completed"] = True
                    _, current_plan["end_time"] = self.environment.get_datetime(
                        format_time=True
                    )
                    related_memories = None
                    if self.params.enable_cognition:
                        try:
                            # Update emotion for the plan
                            related_memories = await self.memory.stream.get_by_ids(
                                current_plan["stream_nodes"]
                            )
                            incident = f"You have successfully completed the plan: {related_memories}"
                            conclusion = (
                                await self.cognition_block.emotion_update(
                                    incident
                                )
                            )
                            await self.save_agent_thought(conclusion)
                            await self.memory.stream.add_cognition_to_memory(
                                current_plan["stream_nodes"], conclusion
                            )
                        except Exception as e:
                            pass
                    await self.memory.status.update("current_plan", current_plan)
                return True
            else:
                # last step is failed
                current_plan["failed"] = True
                _, current_plan["end_time"] = self.environment.get_datetime(
                    format_time=True
                )
                if self.params.enable_cognition:
                    related_memories = None
                    try:
                        # Update emotion for the plan
                        related_memories = await self.memory.stream.get_by_ids(
                            current_plan["stream_nodes"]
                        )
                        incident = (
                            f"You have failed to complete the plan: {related_memories}"
                        )
                        conclusion = (
                            await self.cognition_block.emotion_update(
                                incident
                            )
                        )
                        await self.save_agent_thought(conclusion)
                        await self.memory.stream.add_cognition_to_memory(
                            current_plan["stream_nodes"], conclusion
                        )
                    except Exception as e:
                        pass
                await self.memory.status.update("current_plan", current_plan)
                return True
        # The previous step has not been completed
        return False

    async def process_agent_chat_response(self, payload: dict) -> str:
        """Process incoming social/economic messages and generate responses."""
        if payload["type"] == "social":
            try:
                # Extract basic info
                sender_id = payload.get("from")
                if not sender_id:
                    return ""

                content = payload.get("content", None)

                if not content:
                    return ""

                # add social memory
                description = f"You received a message: {content}"
                await self.memory.stream.add_social(description=description)
                if self.params.enable_cognition:
                    # update emotion
                    await self.cognition_block.emotion_update(description)

                # Get chat histories and ensure proper format
                if "ANNOUNCEMENT" not in content:
                    chat_histories = await self.memory.status.get("chat_histories") or {}
                    if not isinstance(chat_histories, dict):
                        chat_histories = {}

                    # Update chat history with received message
                    if sender_id not in chat_histories:
                        chat_histories[sender_id] = ""
                    if chat_histories[sender_id]:
                        chat_histories[sender_id] += "，"
                    chat_histories[sender_id] += f"them: {content}"

                    # Get relationship score
                    relationships = await self.memory.status.get("relationships") or {}
                    relationship_score = relationships.get(sender_id, 50)

                    # Decision prompt
                    should_respond_prompt = f"""Based on:
- Received message: "{content}"
- Our relationship score: {relationship_score}/100
- My background story: {await self.memory.status.get("background_story")}
- My attitude towards the environmental protection: {await self.memory.status.get("environmental_attitude")}
- My current emotion: {await self.memory.status.get("emotion_types")}
- Recent chat history: {chat_histories.get(sender_id, "")}

Should I respond to this message? Consider:
1. Is this a message that needs/deserves a response? Is this can be the end of the conversation?
2. Would it be natural for someone with my personality to respond?
3. Is our relationship close enough to warrant a response?

Answer only YES or NO, in JSON format, e.g. {{"should_respond": "YES"}}"""

                    should_respond = await self.llm.atext_request(
                        dialog=[
                            {
                                "role": "system",
                                "content": "You are helping decide whether to respond to a message.",
                            },
                            {"role": "user", "content": should_respond_prompt},
                        ],
                        response_format={"type": "json_object"},
                    )
                    should_respond = jsonc.loads(should_respond)["should_respond"]
                    if should_respond == "NO":
                        return ""

                    response_prompt = f"""Based on:
- Received message: "{content}"
- Our relationship score: {relationship_score}/100
- Your background story: {await self.memory.status.get("background_story") or ""}
- Your attitude towards the environmental protection: {await self.memory.status.get("environmental_attitude") or ""}
- My current emotion: {await self.memory.status.get("emotion_types")}
- Recent chat history: {chat_histories.get(sender_id, "")}

Generate an appropriate response that:
1. Matches my personality and background
2. Answer from a first person perspective
3. Is concise (under 100 characters)
4. Reflects our relationship level
5. Ends the conversation when it is natural

Response should be ONLY the message text, no explanations."""

                    response = await self.llm.atext_request(
                        [
                            {
                                "role": "system",
                                "content": "You are helping generate a chat response.",
                            },
                            {"role": "user", "content": response_prompt},
                        ]
                    )

                    if response:
                        # Update chat history with response
                        chat_histories[sender_id] += f", me: {response}"
                        await self.memory.status.update("chat_histories", chat_histories)

                        await self.send_message_to_agent(sender_id, response)
                    return response

            except Exception as e:
                get_logger().warning(f"Error in process_agent_chat_response: {str(e)}")
                return ""
        else:
            content = payload["content"]
            key, value = content.split("@")
            if "." in value:
                value = float(value)
            else:
                value = int(value)
            description = f"You received a economic message: Your {key} has changed from {await self.memory.status.get(key)} to {value}"
            await self.memory.status.update(key, value)
            await self.memory.stream.add_economy(description=description)
            if self.params.enable_cognition:
                await self.cognition_block.emotion_update(description)
            return ""

    async def react_to_intervention(self, intervention_message: str):
        """React to an intervention"""
        # cognition
        conclusion = await self.cognition_block.emotion_update(
            intervention_message
        )
        await self.save_agent_thought(conclusion)
        await self.memory.stream.add_cognition(description=conclusion)
        # needs
        await self.needs_block.reflect_to_intervention(
            intervention_message
        )

    async def reset_position(self):
        """Reset the position of the agent."""
        home = await self.status.get("home")
        home = home["aoi_position"]["aoi_id"]
        await self.environment.reset_person_position(person_id=self.id, aoi_id=home)

    async def step_execution(self):
        """Execute the current step in the active plan based on step type."""
        current_plan = await self.memory.status.get("current_plan")
        if (
            current_plan is None
            or not current_plan
            or len(current_plan.get("steps", [])) == 0
        ):
            return  # No plan, no execution
        step_index = current_plan.get("index", 0)
        execution_context = await self.memory.status.get("execution_context")
        current_step = current_plan.get("steps", [])[step_index]
        # check current_step is valid (not empty)
        if current_step:
            position = await self.memory.status.get("position")
            if "aoi_position" in position:
                current_step["position"] = position["aoi_position"]["aoi_id"]
            current_step["start_time"] = self.environment.get_tick()
            result = None
            dispatch_context = DotDict({"current_intention": current_step["intention"]})
            selected_block = await self.dispatcher.dispatch(dispatch_context)
            if selected_block:
                result = await selected_block.forward(current_step, execution_context)
                if "message" in result:
                    await self.send_message_to_agent(result["target"], result["message"])
            else:
                result = {
                    "success": False,
                    "evaluation": f"Failed to {current_step['intention']}",
                    "consumed_time": 0,
                    "node_id": None,
                }
            if result != None:
                current_step["evaluation"] = result

            # Update current_step, plan, and execution_context information
            current_plan["steps"][step_index] = current_step
            await self.memory.status.update("current_plan", current_plan)
            await self.memory.status.update("execution_context", execution_context)
