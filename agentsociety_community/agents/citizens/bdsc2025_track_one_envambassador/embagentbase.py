from abc import abstractmethod
from typing import List, Dict, Any, Optional
import json
import os

import jsonc
from pydantic import Field
from agentsociety.agent import CitizenAgentBase, AgentToolbox, Block, AgentParams, StatusAttribute
from agentsociety.memory import Memory
from agentsociety.agent import register_get

from .tools import Sence, Poster, Announcement, Communication, MessageProbe
from .fundmanager import FundManager


class EnvAgentBaseConfig(AgentParams):
    ...

class EnvAgentBase(CitizenAgentBase):
    """
    Base class for environmental agents in the competition.
    
    - **Description**:
        - Provides base functionality and tools for environmental agents.
        - Includes tools for sensing, communication, and environmental actions.
        
    - **Args**:
        - `id` (int): The unique identifier for the agent.
        - `name` (str): The name of the agent.
        - `toolbox` (AgentToolbox): The toolbox containing agent utilities.
        - `memory` (Memory): The memory system for the agent.
    """
    ParamsType = EnvAgentBaseConfig
    StatusAttributes = [
        StatusAttribute(name="citizens",type=dict,default={},description="citizens' profile"),
        StatusAttribute(name="citizen_ids",type=list,default=[],description="citizens' ids"),
        StatusAttribute(
            name="probe_logs",
            type=dict,
            default={
                "message": [],
                "poster": [],
                "announcement": []
            },
            description="probe logs"
        ),
        StatusAttribute(name="chat_histories",type=dict,default={},description="all chat histories"),
    ]
    
    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[Any] = None,
        blocks: Optional[list[Block]] = None,
    ) -> None:
        """Initialize the environmental agent with core components and tools."""
        super().__init__(
            id=id,
            name=name,
            toolbox=toolbox,
            memory=memory,
            agent_params=agent_params,
            blocks=blocks,
        )
        
        # Initialize tools
        self._probe = MessageProbe(agent=self, llm=toolbox.llm)
        self.sence = Sence(agent=self, llm=toolbox.llm, probe=self._probe)
        self.poster = Poster(agent=self, llm=toolbox.llm, probe=self._probe)
        self.announcement = Announcement(agent=self, llm=toolbox.llm, probe=self._probe)
        self.communication = Communication(agent=self, llm=toolbox.llm, probe=self._probe)
        
        # Initialize fund manager
        self._fund_manager = FundManager()

    async def before_forward(self):
        """
        Before forward.
        Reset communication times and sync agent status with simulator.
        Do not change anything in this method.
        """
        await super().before_forward()
        # reset communication times
        self.communication._reset()
        # sync agent status with simulator
        await self.update_motion()

    async def reset(self):
        """
        Do not need to reset anything.
        Do not change anything in this method.
        """
        pass

    async def react_to_intervention(self, intervention_message: str):
        """
        Do not need to react to intervention message.
        Do not change anything in this method.
        """
        pass
    
    async def process_agent_chat_response(self, payload: dict) -> str:
        """
        Process incoming messages and generate responses.
        Do not change anything in this method.
        """
        if payload["type"] == "social":
            try:
                sender_id = payload.get("from")
                if not sender_id:
                    return ""
                content = payload.get("content", None)
                if not content:
                    return ""
                chat_histories = await self.memory.status.get("chat_histories", {})
                if sender_id not in chat_histories:
                    chat_histories[sender_id] = f"He/She: {content}"
                else:
                    chat_histories[sender_id] += f"\nHe/She: {content}"
                await self.memory.status.update("chat_histories", chat_histories)
                response = await self.communication_response(sender_id, content)
                if response:
                    await self.communication.sendMessage(sender_id, response)
                return response
            except Exception as e:
                return ""
        else:
            return ""
        
    async def communication_response(self, sender_id: int, content: str):
        """
        Communication response.
        Design your response logic in this method.

        - **Args**:
            - `sender_id` (int): The ID of the sender, agent_id.
            - `content` (str): The content of the message.

        - **Returns**:
            - The response message.
        """
        return "Keep Going!"
        
    async def forward(self):
        """
        Main agent loop.
        Design your own logic in this method.
        """
        pass
