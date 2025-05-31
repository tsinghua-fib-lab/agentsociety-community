import random
import time
from typing import Any, Optional

import jsonc
from agentsociety.agent import (Agent, AgentToolbox, Block, CitizenAgentBase,
                                FormatPrompt, StatusAttribute)
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from agentsociety.memory.const import RelationType, SocialRelation
from agentsociety.message import Message
from agentsociety.survey import Survey

from .blocks import SocialBlock
from .sharing_params import (EnvCitizenBlockOutput, EnvCitizenConfig,
                             EnvCitizenContext)


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


class EnvCitizen(CitizenAgentBase):
    """Agent implementation with configurable cognitive/behavioral modules and social interaction capabilities."""

    ParamsType = EnvCitizenConfig
    BlockOutputType = EnvCitizenBlockOutput
    ContextType = EnvCitizenContext

    StatusAttributes = [
        # Needs Model
        StatusAttribute(
            name="hunger_satisfaction",
            type=float,
            default=0.9,
            description="agent's hunger satisfaction, 0-1",
        ),
        StatusAttribute(
            name="energy_satisfaction",
            type=float,
            default=0.9,
            description="agent's energy satisfaction, 0-1",
        ),
        StatusAttribute(
            name="safety_satisfaction",
            type=float,
            default=0.4,
            description="agent's safety satisfaction, 0-1",
        ),
        StatusAttribute(
            name="social_satisfaction",
            type=float,
            default=0.6,
            description="agent's social satisfaction, 0-1",
        ),
        StatusAttribute(
            name="current_need",
            type=str,
            default="none",
            description="agent's current need",
        ),
        # Plan Behavior Model
        StatusAttribute(
            name="current_plan",
            type=dict,
            default={},
            description="agent's current plan",
        ),
        StatusAttribute(
            name="execution_context",
            type=dict,
            default={},
            description="agent's execution context",
        ),
        StatusAttribute(
            name="plan_history",
            type=list,
            default=[],
            description="agent's plan history",
        ),
        # Cognition
        StatusAttribute(
            name="emotion",
            type=dict,
            default={
                "sadness": 5,
                "joy": 5,
                "fear": 5,
                "disgust": 5,
                "anger": 5,
                "surprise": 5,
            },
            description="agent's emotion, 0-10",
        ),
        StatusAttribute(
            name="attitude",
            type=dict,
            default={},
            description="agent's attitude",
            whether_embedding=True,
        ),
        StatusAttribute(
            name="thought",
            type=str,
            default="Currently nothing good or bad is happening",
            description="agent's thought",
            whether_embedding=True,
        ),
        StatusAttribute(
            name="emotion_types",
            type=str,
            default="Relief",
            description="agent's emotion types",
            whether_embedding=True,
        ),
        # Economy
        StatusAttribute(
            name="work_skill", type=float, default=0.0, description="agent's work skill"
        ),
        StatusAttribute(
            name="tax_paid", type=float, default=0.0, description="agent's tax paid"
        ),
        StatusAttribute(
            name="consumption_currency",
            type=float,
            default=0.0,
            description="agent's consumption currency",
        ),
        StatusAttribute(
            name="goods_demand", type=int, default=0, description="agent's goods demand"
        ),
        StatusAttribute(
            name="goods_consumption",
            type=int,
            default=0,
            description="agent's goods consumption",
        ),
        StatusAttribute(
            name="work_propensity",
            type=float,
            default=0.0,
            description="agent's work propensity",
        ),
        StatusAttribute(
            name="consumption_propensity",
            type=float,
            default=0.0,
            description="agent's consumption propensity",
        ),
        StatusAttribute(
            name="to_consumption_currency",
            type=float,
            default=0.0,
            description="agent's to consumption currency",
        ),
        StatusAttribute(
            name="firm_id", type=int, default=0, description="agent's firm id"
        ),
        StatusAttribute(
            name="government_id",
            type=int,
            default=0,
            description="agent's government id",
        ),
        StatusAttribute(
            name="bank_id", type=int, default=0, description="agent's bank id"
        ),
        StatusAttribute(
            name="nbs_id", type=int, default=0, description="agent's nbs id"
        ),
        StatusAttribute(
            name="dialog_queue",
            type=list,
            default=[],
            description="agent's dialog queue",
        ),
        StatusAttribute(
            name="firm_forward", type=int, default=0, description="agent's firm forward"
        ),
        StatusAttribute(
            name="bank_forward", type=int, default=0, description="agent's bank forward"
        ),
        StatusAttribute(
            name="nbs_forward", type=int, default=0, description="agent's nbs forward"
        ),
        StatusAttribute(
            name="government_forward",
            type=int,
            default=0,
            description="agent's government forward",
        ),
        StatusAttribute(
            name="forward", type=int, default=0, description="agent's forward"
        ),
        StatusAttribute(
            name="depression",
            type=float,
            default=0.0,
            description="agent's depression, 0-1",
        ),
        StatusAttribute(
            name="ubi_opinion", type=list, default=[], description="agent's ubi opinion"
        ),
        StatusAttribute(
            name="working_experience",
            type=list,
            default=[],
            description="agent's working experience",
        ),
        StatusAttribute(
            name="work_hour_month",
            type=float,
            default=160,
            description="agent's work hour per month",
        ),
        StatusAttribute(
            name="work_hour_finish",
            type=float,
            default=0,
            description="agent's work hour finished",
        ),
        # Social
        StatusAttribute(
            name="friends_info",
            type=dict,
            default={},
            description="agent's friends info",
        ),
        StatusAttribute(
            name="relationships",
            type=dict,
            default={},
            description="agent's relationship strength with each friend",
        ),
        StatusAttribute(
            name="relation_types",
            type=dict,
            default={},
            description="agent's relation types with each friend",
        ),
        StatusAttribute(
            name="chat_histories",
            type=dict,
            default={},
            description="all chat histories",
        ),
        StatusAttribute(
            name="interactions",
            type=dict,
            default={},
            description="all interaction records",
        ),
        # Mobility
        StatusAttribute(
            name="number_poi_visited",
            type=int,
            default=1,
            description="agent's number of poi visited",
        ),
        StatusAttribute(
            name="location_knowledge",
            type=dict,
            default={},
            description="agent's location knowledge",
        ),
        StatusAttribute(
            name="message_propagation_preference",
            type=str,
            default="",
            description="agent's message propagation preference",
        ),
        StatusAttribute(
            name="background_story",
            type=str,
            default="",
            description="agent's background story",
        ),
    ]

    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[EnvCitizenConfig] = None,
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
        self.social_block = SocialBlock(
            agent=self,
            llm=self.llm,
            max_visible_followers=self.params.max_visible_followers,
            max_private_chats=self.params.max_private_chats,
            chat_probability=self.params.chat_probability,
            environment=self.environment,
            memory=self.memory,
        )
        self.step_count = -1

    async def before_forward(self):
        """Before forward"""
        await super().before_forward()

    async def reset(self):
        """Reset the agent."""
        # reset position to home
        await self.reset_position()

        # reset needs
        await self.memory.status.update("current_need", "none")

        # reset plans and actions
        await self.memory.status.update("current_plan", {})
        await self.memory.status.update("execution_context", {})

    # Main workflow
    async def forward(
        self,
    ):
        """Main agent loop coordinating status updates, plan execution, and cognition."""
        start_time = time.time()
        self.step_count += 1
        # sync agent status with simulator
        await self.update_motion()
        get_logger().debug(f"Agent {self.id}: Finished main workflow - update motion")

        # ATTENTION: random social interaction
        current_messages = await self.social_block.current_messages()
        received_ids = set(ii for (ii, _) in current_messages)
        if (
            random.random() < self.params.chat_probability
            or self.params.rumor_post_identifier in received_ids
        ):
            # social interaction
            await self.social_block.forward(None)
            get_logger().debug(
                f"Agent {self.id}: Finished main workflow - social interaction"
            )

        get_logger().debug(f"Agent {self.id}: Finished main workflow - cognition")

        return time.time() - start_time

    async def do_chat(self, message: Message) -> str:
        """Process incoming social/economic messages and generate responses."""
        payload = message.payload
        if payload["type"] == "social":
            resp = f"Agent {self.id} received agent chat response: {payload}"
            try:
                # Extract basic info
                sender_id = payload.get("from")
                if not sender_id:
                    return ""

                raw_content = payload.get("content", "")

                # Parse message content
                try:
                    message_data = jsonc.loads(raw_content)
                    content = message_data["content"]
                    propagation_count = message_data.get("propagation_count", 1)
                except (jsonc.JSONDecodeError, TypeError, KeyError):
                    content = raw_content
                    propagation_count = 1

                if not content:
                    return ""

                await self.social_block.receive_message(sender_id, f"{content}")

                # add social memory
                description = f"You received a social message: {content}"
                await self.memory.stream.add_social(description=description)
            except Exception as e:
                get_logger().warning(f"Error in process_agent_chat_response: {str(e)}")
                return ""
        elif payload["type"] == "persuasion":
            content = payload["content"]
            # add persuasion memory
            description = f"You received a persuasion message: {content}"
            await self.memory.stream.add_social(description=description)
            await self.social_block._add_intervention_to_history(
                intervention_type="persuasion_received",
                details={
                    "content": content,
                },
            )
        elif payload["type"] == "remove-follower":
            to_remove_id = payload["to_remove_id"]
            current_social_network: list[SocialRelation] = await self.memory.status.get(
                "social_network"
            )
            current_social_network = [
                connection
                for connection in current_social_network
                if connection.target_id != to_remove_id
                and connection.kind == RelationType.FOLLOWER
            ]
            await self.memory.status.update("social_network", current_social_network)
            await self.social_block._add_intervention_to_history(
                intervention_type="remove_follower_by_platform",
                details={
                    "to_remove_id": to_remove_id,
                },
            )
        elif payload["type"] == "remove-following":
            to_remove_id = payload["to_remove_id"]
            current_social_network: list[SocialRelation] = await self.memory.status.get(
                "social_network"
            )
            current_social_network = [
                connection
                for connection in current_social_network
                if connection.target_id != to_remove_id
                and connection.kind == RelationType.FOLLOWING
            ]
            await self.memory.status.update("social_network", current_social_network)
            await self.social_block._add_intervention_to_history(
                intervention_type="remove_following_by_platform",
                details={
                    "to_remove_id": to_remove_id,
                },
            )
        elif payload["type"] == "agent_banned":
            await self.social_block._add_intervention_to_history(
                intervention_type="agent_banned",
                details={},
            )
        elif payload["type"] == "post_deleted":
            await self.social_block._add_intervention_to_history(
                intervention_type="post_deleted",
                details={
                    "post_id": payload["post_id"],
                },
            )
        return ""

        # async def do_survey(self, survey: Survey) -> str:
        """
        Generate a response to a user survey based on the agent's memory and current state.

        - **Args**:
            - `survey` (`Survey`): The survey that needs to be answered.

        - **Returns**:
            - `str`: The generated response from the agent.

        - **Description**:
            - Prepares a prompt for the Language Model (LLM) based on the provided survey.
            - Constructs a dialog including system prompts, relevant memory context, and the survey question itself.
            - Uses the LLM client to generate a response asynchronously.
            - If the LLM client is not available, it returns a default message indicating unavailability.
            - This method can be overridden by subclasses to customize survey response generation.
        """
        survey_prompt = survey.to_prompt()
        dialog = []

        # Add system prompt
        system_prompt = "Please answer the survey question in first person. Follow the format requirements strictly and provide clear and specific answers (In JSON format)."
        dialog.append({"role": "system", "content": system_prompt})

        # Add memory context
        if self.memory:
            message_summary = self.social_block.history_summary
            preference = await self.memory.status.get(
                "message_propagation_preference", ""
            )
            dialog.append(
                {
                    "role": "system",
                    "content": f"你最终了解到的外部信息总结如下：\n{message_summary}\n \n{self.social_block.preference_appendix.get(preference, '')}\n",
                }
            )

        # Add survey question
        dialog.append({"role": "user", "content": survey_prompt})

        for retry in range(10):
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
            get_logger().error("Failed to generate survey response")
            json_str = ""
        return json_str

    async def do_survey(self, survey: Survey) -> str:
        """
        Generate a response to a user survey based on the agent's memory and current state.

        - **Args**:
            - `survey` (`Survey`): The survey that needs to be answered.

        - **Returns**:
            - `str`: The generated response from the agent.

        - **Description**:
            - Prepares a prompt for the Language Model (LLM) based on the provided survey.
            - Constructs a dialog including system prompts, relevant memory context, and the survey question itself.
            - Uses the LLM client to generate a response asynchronously.
            - If the LLM client is not available, it returns a default message indicating unavailability.
            - This method can be overridden by subclasses to customize survey response generation.
        """
        survey_prompt = survey.to_prompt()
        dialog = []

        # Add system prompt
        system_prompt = "Please answer the survey question in first person. Follow the format requirements strictly and provide clear and specific answers (In JSON format)."
        dialog.append({"role": "system", "content": system_prompt})

        # Add memory context
        if self.memory:
            message_summary = self.social_block.history_summary
            preference = await self.memory.status.get(
                "message_propagation_preference", ""
            )
            dialog.append(
                {
                    "role": "system",
                    "content": f"你最终了解到的外部信息总结如下：\n{message_summary}\n \n{self.social_block.preference_appendix.get(preference, '')}\n",
                }
            )

        # Add survey question
        dialog.append({"role": "user", "content": survey_prompt})

        for retry in range(10):
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
            get_logger().error("Failed to generate survey response")
            json_str = ""
        return json_str

    async def react_to_intervention(self, intervention_message: str):
        """React to an intervention"""
        pass

    async def reset_position(self):
        """Reset the position of the agent."""
        home = await self.status.get("home")
        home = home["aoi_position"]["aoi_id"]
        await self.environment.reset_person_position(person_id=self.id, aoi_id=home)

    async def close(self):
        """Close the agent."""
        pass
