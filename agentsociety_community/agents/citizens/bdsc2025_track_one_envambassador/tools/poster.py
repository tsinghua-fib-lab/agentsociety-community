"""
Poster tool for the Agent Society.
"""

from typing import Optional, Union
from agentsociety.agent import Agent
from agentsociety.llm import LLM
from .message_probe import MessageProbe

POSTER_COST = 3000
POSTER_EMBELLISHMENT = """
========================================
THIS IS A POSTER PUT UP BY THE ENVIRONMENT PROTECTION DEPARTMENT.
========================================
"""


class Poster:
    """
    Tool for posting notices in specific areas.
    
    - **Description**:
        - Provides methods for agents to put up posters in specific areas of interest.
        
    - **Args**:
        - `agent` (Agent): The agent using this tool.
        - `llm` (LLM): The language model used by the agent.
        - `probe` (MessageProbe): The message evaluation tool.
    """
    
    def __init__(self, agent: Agent, llm: LLM, probe: MessageProbe):
        self.agent = agent
        self.llm = llm
        self.probe = probe
        
    async def putUpPoster(self, target_aoi_ids: Union[list[int], int], content: str, reason: Optional[str] = None):
        """
        Puts up a poster in specific areas of interest.
        Each poster costs 3000 units of funds (for each aoi).
        
        - **Description**:
            - Places a poster with the specified content in the specified areas of interest.
            - Citizens in or passing through these areas will be able to "discover" the poster.
            - Citizens will not respond to the poster.
            
        - **Args**:
            - `target_aoi_ids` (List[int]): The IDs of the areas of interest to put the poster in.
            - `content` (str): The content of the poster.
            
        - **Returns**:
            - None
        """
        if isinstance(target_aoi_ids, int):
            target_aoi_ids = [target_aoi_ids]
        length = len(target_aoi_ids)
        success = await self.agent._fund_manager.update_funds(POSTER_COST * length, reason) # type: ignore
        if not success:
            return {
                "success": False,
                "reason": f"You don't have enough funds to put up poster for {length} aois."
            }
        await self.probe.probePoster(content)
        await self.agent.register_aoi_message(target_aoi_ids, POSTER_EMBELLISHMENT + content)
        return {
            "success": True,
            "reason": f"You have put up poster for {length} aois."
        }