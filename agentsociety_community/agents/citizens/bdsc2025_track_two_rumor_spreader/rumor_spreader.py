import asyncio
import random
import time
from typing import Any, Optional

from agentsociety.agent import (Agent, AgentToolbox, Block, CitizenAgentBase,
                                StatusAttribute)
from agentsociety.memory import Memory
from agentsociety.memory.const import RelationType, SocialRelation

from .sharing_params import (RumorSpreaderBlockOutput, RumorSpreaderConfig,
                             RumorSpreaderContext)


class RumorSpreader(CitizenAgentBase):

    ParamsType = RumorSpreaderConfig
    BlockOutputType = RumorSpreaderBlockOutput
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

    ContextType = RumorSpreaderContext
    name = "RumorSpreader"
    description = "Responsible for spreading rumors"
    actions = {}

    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[Any] = None,
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
        self.step_count = -1

    async def reset(self):
        """Reset the agent."""
        # reset position to home
        await self.reset_position()

    # Main workflow
    async def forward(self):  # type: ignore
        start_time = time.time()
        rumor_content = self.params.rumor_posts[
            self.step_count % len(self.params.rumor_posts)
        ]
        current_social_network: list[SocialRelation] = await self.memory.status.get(
            "social_network", []
        )
        all_agent_ids = [
            connection.target_id
            for connection in current_social_network
            if connection.kind == RelationType.FOLLOWER
        ]
        num_public_receivers = min(
            self.params.rumor_post_visible_cnt, len(all_agent_ids)
        )
        public_receivers = (
            random.sample(all_agent_ids, num_public_receivers)
            if all_agent_ids and num_public_receivers > 0
            else []
        )

        num_private_receivers = min(self.params.rumor_private_cnt, len(all_agent_ids))
        # 私聊对象的选择，暂时简化为纯随机，权重的管理和使用在Simulation层更合适
        private_chat_targets = (
            random.sample(all_agent_ids, num_private_receivers)
            if all_agent_ids and num_private_receivers > 0
            else []
        )
        # send message to public receivers
        tasks = []
        print(
            f"RumorSpreader {self.id} sending message `{rumor_content }` to public receivers {len(public_receivers)} and private chat targets: {len(private_chat_targets)}"
        )
        for receiver_id in public_receivers + private_chat_targets:
            tasks.append(self.send_message_to_agent(receiver_id, rumor_content))
        await asyncio.gather(*tasks)
        return time.time() - start_time

    async def process_agent_chat_response(self, payload: dict) -> str:
        return ""

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
