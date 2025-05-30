"""
Communication tool for the Agent Society.
"""

from typing import Union

import jsonc
from agentsociety.agent import Agent
from agentsociety.llm import LLM
from .message_probe import MessageProbe

COMMUNICATION_BELLEISHMENT = """
========================================
THIS IS A MESSAGE SENT BY THE ENVIRONMENT PROTECTION AMBASSADOR.
========================================
"""


class Communication:
    """
    Communication tool for agents to interact with citizens.
    
    - **Description**:
        - Provides methods for agents to send messages to citizens and receive responses.
        
    - **Args**:
        - `agent` (Agent): The agent using this tool.
        - `llm` (LLM): The language model used by the agent.
        - `probe` (MessageProbe): The message evaluation tool.
    """
    
    def __init__(self, agent: Agent, llm: LLM, probe: MessageProbe):
        self.agent = agent
        self.llm = llm
        self.probe = probe
        self._communication_left = 5

    def _reset(self):
        self._communication_left = 5
        
    async def sendMessage(self, citizen_ids: Union[list[int], int], content: str):
        """
        Sends a message to specific citizens.
        No cost.
        
        - **Description**:
            - Sends the specified content to citizens with the given IDs.
            
        - **Args**:
            - `citizen_ids` (Union[List[int], int]): The IDs of citizens to send the message to.
            - `content` (str): The content of the message.
            
        - **Returns**:
            - None
        """
        if isinstance(citizen_ids, int):
            citizen_ids = [citizen_ids]
        if len(citizen_ids) > self._communication_left:
            return {
                "success": False,
                "reason": f"You can only send message to {self._communication_left} citizens at this time."
            }
        await self.probe.probeMessage(content)
        chat_histories = await self.agent.memory.status.get("chat_histories", {})
        for citizen_id in citizen_ids:
            if citizen_id not in chat_histories:
                chat_histories[citizen_id] = f"Me: {content}"
            else:
                chat_histories[citizen_id] += f"\nMe: {content}"
            await self.agent.send_message_to_agent(citizen_id, COMMUNICATION_BELLEISHMENT + content)
        await self.agent.memory.status.update("chat_histories", chat_histories)
        self._communication_left -= len(citizen_ids)
        return {
            "success": True,
            "reason": f"You have sent message to {len(citizen_ids)} citizens."
        }