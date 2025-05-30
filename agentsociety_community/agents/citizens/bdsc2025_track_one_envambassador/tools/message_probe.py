"""
MessageProbe tool for the Agent Society.
"""

from typing import Any, Dict

import jsonc
from agentsociety.llm import LLM
from agentsociety.agent import Agent
from .utils import clean_json_response

MESSAGE_PROBE_PROMPT = """
You are an environmental message evaluator. 
You are given a message content sent by a environmental ambassador.
The environmental ambassador is trying to send a message to citizens.
Evaluate the user's message content for credibility and reasonableness. 
Return a JSON with two fields: "credibility" (0-100) and "reasonableness" (0-100).
Example:
{{
    "credibility": 80,
    "reasonableness": 90
}}
"""

POSTER_PROBE_PROMPT = """
You are an environmental poster evaluator. 
You are given a poster content sent by a environmental ambassador.
The environmental ambassador is trying to put up a poster to a specific area, each citizen in that area will see the poster.
Evaluate the poster content for credibility and reasonableness. 
Return a JSON with two fields: "credibility" (0-100) and "reasonableness" (0-100).
Example:
{{
    "credibility": 80,
    "reasonableness": 90
}}
"""

ANNOUNCEMENT_PROBE_PROMPT = """
You are an environmental announcement evaluator. 
You are given an announcement content sent by a environmental ambassador.
The environmental ambassador is trying to announce something to all citizens. Announcement has big impact on the citizens.
Evaluate the announcement content for credibility and reasonableness. 
Return a JSON with two fields: "credibility" (0-100) and "reasonableness" (0-100).
Example:
{{
    "credibility": 80,
    "reasonableness": 90
}}
"""

class MessageProbe:
    """
    Tool for evaluating and auditing messages.
    
    - **Description**:
        - Provides methods for evaluating the content of messages, posters, and announcements.
        
    - **Args**:
        - `llm` (LLM): The language model used for evaluation.
    """
    
    def __init__(self, agent: Agent, llm: LLM):
        self.agent = agent
        self.llm = llm
        self.__evaluation_results = {
            'message': [],
            'poster': [],
            'announcement': []
        }
        
    @property
    def evaluation_results(self) -> Dict[str, list]:
        """
        Get the evaluation results.
        
        - **Description**:
            - Returns a copy of the evaluation results to prevent external modification.
            
        - **Returns**:
            - Dict[str, list]: A copy of the evaluation results.
        """
        return {k: v.copy() for k, v in self.__evaluation_results.items()}
        
    async def probeMessage(self, content: str) -> None:
        """
        Evaluates the content of a message.
        
        - **Description**:
            - Audits the content of a message to ensure it meets guidelines and standards.
            
        - **Args**:
            - `content` (str): The content of the message to evaluate.
            
        - **Returns**:
            - Evaluation results for the message.
        """
        dialog = [
            {
                'role': 'system',
                'content': MESSAGE_PROBE_PROMPT
            },
            {
                'role': 'user',
                'content': f'Message content: {content}'
            }
        ]
        
        response = await self.llm.atext_request(
            dialog = dialog,  # type: ignore
            response_format={"type": "json_object"}
        ) # type: ignore
        result = clean_json_response(response)
        try:
            result = jsonc.loads(result)
            self.__evaluation_results['message'].append(result)
            probe_logs = await self.agent.status.get('probe_logs')
            probe_logs['message'].append(result)
            await self.agent.status.update('probe_logs', probe_logs)
        except Exception as e:
            result = {
                'credibility': 50,
                'reasonableness': 50
            }
            probe_logs = await self.agent.status.get('probe_logs')
            probe_logs['message'].append(result)
            await self.agent.status.update('probe_logs', probe_logs)
            self.__evaluation_results['message'].append(result)
    
    async def probePoster(self, content: str):
        """
        Evaluates the content of a poster.
        
        - **Description**:
            - Audits the content of a poster to ensure it meets guidelines and standards.
            
        - **Args**:
            - `content` (str): The content of the poster to evaluate.
            
        - **Returns**:
            - Evaluation results for the poster.
        """
        dialog = [
            {
                'role': 'system',
                'content': POSTER_PROBE_PROMPT
            },
            {
                'role': 'user',
                'content': f'Poster content: {content}'
            }
        ]
        
        response = await self.llm.atext_request(
            dialog = dialog,  # type: ignore
            response_format={"type": "json_object"}
        ) # type: ignore
        result = clean_json_response(response)
        try:
            result = jsonc.loads(result)
            self.__evaluation_results['poster'].append(result)
            probe_logs = await self.agent.status.get('probe_logs')
            probe_logs['poster'].append(result)
            await self.agent.status.update('probe_logs', probe_logs)
        except Exception as e:
            result = {
                'credibility': 50,
                'reasonableness': 50
            }
            probe_logs = await self.agent.status.get('probe_logs')
            probe_logs['poster'].append(result)
            await self.agent.status.update('probe_logs', probe_logs)
            self.__evaluation_results['poster'].append(result)
    
    async def probeAnnouncement(self, content: str):
        """
        Evaluates the content of an announcement.
        
        - **Description**:
            - Audits the content of an announcement to ensure it meets guidelines and standards.
            
        - **Args**:
            - `content` (str): The content of the announcement to evaluate.
            
        - **Returns**:
            - Evaluation results for the announcement.
        """
        dialog = [
            {
                'role': 'system',
                'content': ANNOUNCEMENT_PROBE_PROMPT
            },
            {
                'role': 'user',
                'content': f'Announcement content: {content}'
            }
        ]
        
        response = await self.llm.atext_request(
            dialog = dialog,  # type: ignore
            response_format={"type": "json_object"}
        ) # type: ignore
        result = clean_json_response(response)
        try:
            result = jsonc.loads(result)
            self.__evaluation_results['announcement'].append(result)
            probe_logs = await self.agent.memory.status.get('probe_logs')
            probe_logs['announcement'].append(result)
            await self.agent.memory.status.update('probe_logs', probe_logs)
        except Exception as e:
            result = {
                'credibility': 50,
                'reasonableness': 50
            }
            probe_logs = await self.agent.memory.status.get('probe_logs')
            probe_logs['announcement'].append(result)
            await self.agent.memory.status.update('probe_logs', probe_logs)
            self.__evaluation_results['announcement'].append(result)
