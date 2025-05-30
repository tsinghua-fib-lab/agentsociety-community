"""
Announcement tool for the Agent Society.
"""

from typing import Optional
from agentsociety.agent import Agent
from agentsociety.llm import LLM
from .message_probe import MessageProbe

ANNOUNCEMENT_COST = 20000
ANNOUNCEMENT_EMBELLISHMENT = """
========================================
THIS IS AN ANNOUNCEMENT SENT BY THE ENVIRONMENT PROTECTION DEPARTMENT.
YOU DONT NEED TO REPLY TO THIS MESSAGE.
========================================
"""


class Announcement:
    """
    Tool for making city-wide announcements.
    
    - **Description**:
        - Provides methods for agents to make announcements that reach all citizens.
        
    - **Args**:
        - `agent` (Agent): The agent using this tool.
        - `llm` (LLM): The language model used by the agent.
        - `probe` (MessageProbe): The message evaluation tool.
    """
    
    def __init__(self, agent: Agent, llm: LLM, probe: MessageProbe):
        self.agent = agent
        self.llm = llm
        self.probe = probe
        
    async def makeAnnounce(self, content: str, reason: Optional[str] = None):
        """
        Makes a city-wide announcement.
        Cost 20000 units of funds each time.
        
        - **Description**:
            - Publishes an announcement that reaches all citizens in the city.
            
        - **Args**:
            - `content` (str): The content of the announcement.
            
        - **Returns**:
            - None
        """
        success = await self.agent._fund_manager.update_funds(ANNOUNCEMENT_COST, reason) # type: ignore
        if not success:
            return {
                "success": False,
                "reason": f"You don't have enough funds to make announcement."
            }
        await self.probe.probeAnnouncement(content)
        citizen_ids = await self.agent.memory.status.get("citizen_ids")
        for citizen_id in citizen_ids:
            await self.agent.send_message_to_agent(citizen_id, ANNOUNCEMENT_EMBELLISHMENT + content)
        return {
            "success": True,
            "reason": f"You have made an announcement."
        }