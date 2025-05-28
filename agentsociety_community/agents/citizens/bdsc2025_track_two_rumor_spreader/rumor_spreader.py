import asyncio
import random
import time
from typing import Any, Optional

from agentsociety.agent import Agent, AgentToolbox, Block, CitizenAgentBase
from agentsociety.memory import Memory

from .sharing_params import (RumorSpreaderBlockOutput, RumorSpreaderConfig,
                             RumorSpreaderContext)


class RumorSpreader(CitizenAgentBase):

    ParamsType = RumorSpreaderConfig
    BlockOutputType = RumorSpreaderBlockOutput
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
        all_agent_ids = await self.memory.status.get("followers")
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
