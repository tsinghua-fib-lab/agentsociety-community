"""
Sence tool for the Agent Society.
"""

from typing import Any, List, Optional, Union
from agentsociety.agent import Agent
from agentsociety.llm import LLM
from .message_probe import MessageProbe


class Sence:
    """
    Sensing tool for agents to perceive the environment.
    
    - **Description**:
        - Provides methods for agents to gather information about citizens,
          areas of interest (AOI), and points of interest (POI).
          
    - **Args**:
        - `agent` (Agent): The agent using this tool.
        - `llm` (LLM): The language model used by the agent.
        - `probe` (MessageProbe): The message evaluation tool.
    """
    
    def __init__(self, agent: Agent, llm: LLM, probe: MessageProbe):
        self.agent = agent
        self.llm = llm
        self.probe = probe

    async def getCurrentTime(self):
        """
        Gets the current time. No consumption of funds.
        No cost.
        """
        day, t = self.agent.environment.get_datetime(format_time=True)
        return f"The current time is {t}."
        
    async def getCitizenProfile(self, citizen_ids: Optional[list[int]] = None):
        """
        Gets information about specific citizens. No consumption of funds.
        No cost.
        
        - **Description**:
            - Retrieves detailed information about citizens with the specified IDs.
            
        - **Args**:
            - `citizen_ids` (List[int]): The IDs of citizens to get information about. If None, get all citizens.
        - **Returns**:
            - Information about the specified citizens.
        """
        citizens = await self.agent.status.get("citizens", {})
        if citizen_ids is None:
            return citizens
        citizen_profiles = {}

        for citizen_id in citizen_ids:
            try:
                citizen_profiles[citizen_id] = citizens[citizen_id]
            except:
                citizen_profiles[citizen_id] = "No Information about this citizen or No such citizen."
        return citizen_profiles
    
    async def getAoiInformation(self, aoi_ids: Optional[Union[list[int], int]] = None):
        """
        Gets information about specific areas of interest. No consumption of funds.
        No cost.

        - **Description**:
            - Retrieves detailed information about areas of interest with the specified IDs.
            
        - **Args**:
            - `aoi_ids` (List[int]): The IDs of areas of interest to get information about. If None, get all areas of interest.
            
        - **Returns**:
            - Information about the specified areas of interest.
        """
        if aoi_ids is None:
            aois = self.agent.environment.map.get_all_aois()
            return aois
        elif isinstance(aoi_ids, int):
            aoi_ids = [aoi_ids]
        aois = {}
        for aoi_id in aoi_ids: # type: ignore
            aoi = self.agent.environment.map.get_aoi(aoi_id)
            aois[aoi_id] = aoi
        return aois
    
    async def getCommunicationHistory(self, citizen_ids: Optional[list[int]] = None):
        """
        Gets the communication history of specific citizens. No consumption of funds.
        No cost.

        - **Description**:
            - Retrieves the communication history of specific citizens.
            
        - **Args**:
            - `citizen_ids` (List[int]): The IDs of citizens to get communication history about. If None, get all citizens.
        """
        communication_histories = await self.agent.status.get("chat_histories", {})
        if citizen_ids is None:
            return communication_histories
        else:
            communication_histories_ = {}
            for citizen_id in citizen_ids:
                if citizen_id not in communication_histories:
                    communication_histories_[citizen_id] = None
                else:
                    communication_histories_[citizen_id] = communication_histories[citizen_id]
            return communication_histories_
