import asyncio
import logging
import numbers
import random
from typing import Optional

import jsonc
import numpy as np
from pydantic import Field

from agentsociety.agent import Block, FormatPrompt, BlockParams, BlockDispatcher, DotDict
from agentsociety.environment import EconomyClient, Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from .utils import *


def softmax(x, gamma=1.0):
    """Compute softmax values with temperature scaling.

    Args:
        x: Input values (array or list)
        gamma: Temperature parameter (higher values make distribution sharper)

    Returns:
        Probability distribution over input values
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x *= gamma
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class WorkBlock(Block):
    """Handles work-related economic activities and time tracking.

    Attributes:
        guidance_prompt: Template for time estimation queries
    """
    name = "WorkBlock"
    description = "Handles work-related economic activities and time tracking"

    def __init__(self, llm: LLM, environment: Environment, memory: Memory, worktime_estimation_prompt: str = TIME_ESTIMATE_PROMPT):
        """Initialize with dependencies.

        Args:
            llm: Language model for time estimation
            environment: Time tracking environment
            memory: Agent's memory system
        """
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=memory,
        )
        self.guidance_prompt = FormatPrompt(template=worktime_estimation_prompt)

    async def forward(self, step, plan_context):
        """Process work task and track time expenditure.

        Workflow:
            1. Format prompt with work context
            2. Request time estimation from LLM
            3. Record work experience in memory
            4. Fallback to random time on parsing failures

        Returns:
            Execution result with time consumption details
        """
        node_id = await self.memory.stream.add_economy(
            description=f"I worked.({step['intention']})"
        )
        return {
            "success": True,
            "evaluation": f'Finished {step["intention"]}',
            "consumed_time": random.randint(10, 120),
            "node_id": node_id,
        }
        await self.guidance_prompt.format(
            plan=plan_context["plan"],
            intention=step["intention"],
            emotion_types=await self.memory.status.get("emotion_types"),
        )
        result = await self.llm.atext_request(
            self.guidance_prompt.to_dialog(), response_format={"type": "json_object"}
        )
        result = clean_json_response(result)
        try:
            result = jsonc.loads(result)
            time = result["time"]
            day, start_time = self.environment.get_datetime(format_time=True)
            await self.memory.status.update(
                "working_experience",
                [
                    f"Start from {start_time}, worked {time} minutes on {step['intention']}"
                ],
                mode="merge",
            )
            work_hour_finish = await self.memory.status.get("work_hour_finish")
            work_hour_finish += float(time / 60)
            node_id = await self.memory.stream.add_economy(
                description=f"I worked {time} minutes on {step['intention']}"
            )
            await self.memory.status.update("work_hour_finish", work_hour_finish)
            return {
                "success": True,
                "evaluation": f'work: {step["intention"]}',
                "consumed_time": time,
                "node_id": node_id,
            }
        except Exception as e:
            get_logger().warning(
                f"Error in parsing: {str(e)}, raw: {result}"
            )
            time = random.randint(1, 3) * 60
            day, start_time = self.environment.get_datetime(format_time=True)
            await self.memory.status.update(
                "working_experience",
                [
                    f"Start from {start_time}, worked {time} minutes on {step['intention']}"
                ],
                mode="merge",
            )
            work_hour_finish = await self.memory.status.get("work_hour_finish")
            node_id = await self.memory.stream.add_economy(
                description=f"I worked {time} minutes on {step['intention']}"
            )
            work_hour_finish += float(time / 60)
            await self.memory.status.update("work_hour_finish", work_hour_finish)
            return {
                "success": True,
                "evaluation": f'work: {step["intention"]}',
                "consumed_time": time,
                "node_id": node_id,
            }


class ConsumptionBlock(Block):
    """Manages consumption behavior and budget constraints.

    Attributes:
        economy_client: Interface to economic simulation
        forward_times: Counter for execution attempts
    """
    name = "ConsumptionBlock"
    description = "Used to determine the consumption amount, and items"

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
    ):
        """Initialize consumption processor.

        Args:
            economy_client: Client for economic system interactions
        """
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=agent_memory,
        )
        self.forward_times = 0

    async def forward(self, step, plan_context):
        """Execute consumption decision-making.

        Workflow:
            1. Check monthly consumption limits
            2. Calculate price-weighted demand distribution
            3. Execute transactions through economy client
            4. Record consumption in memory stream

        Returns:
            Consumption evaluation with financial details
        """
        self.forward_times += 1
        node_id = await self.memory.stream.add_economy(
            description=f"I bought some goods on {step['intention']}"
        )
        evaluation = {
            "success": True,
            "evaluation": f"I bought some goods on {step['intention']}",
            "consumed_time": 20,
            "node_id": node_id,
        }
        return evaluation


class EconomyNoneBlock(Block):
    """
    Fallback block for non-economic/non-specified activities.
    """
    name = "EconomyNoneBlock"
    description = "Fallback block for other activities"

    def __init__(self, llm: LLM, memory: Memory):
        super().__init__(
            llm=llm, agent_memory=memory
        )

    async def forward(self, step, plan_context):
        """Log generic activities in economy stream."""
        node_id = await self.memory.stream.add_economy(
            description=f"I {step['intention']}"
        )
        return {
            "success": True,
            "evaluation": f'Finished{step["intention"]}',
            "consumed_time": 0,
            "node_id": node_id,
        }
    
class EnvEconomyBlockParams(BlockParams):
    worktime_estimation_prompt: str = Field(default=TIME_ESTIMATE_PROMPT, description="Used to determine the worktime")

class EnvEconomyBlock(Block):
    """Orchestrates economic activities through specialized sub-blocks.

    Attributes:
        dispatcher: Routes tasks to appropriate sub-blocks
        work_block: Work activity handler
        consumption_block: Consumption manager
        none_block: Fallback activities
    """
    ParamsType = EnvEconomyBlockParams
    name = "EconomyBlock"
    description = "Orchestrates economic activities through specialized actions"
    actions = {
        "work": "Support the work action",
        "consume": "Support the consume action",
        "economy_none": "Support other economic operations",
    }

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        block_params: Optional[EnvEconomyBlockParams] = None,
    ):
        super().__init__(
            llm=llm, environment=environment, agent_memory=agent_memory, block_params=block_params
        )
        self.work_block = WorkBlock(llm, environment, agent_memory, self.params.worktime_estimation_prompt)
        self.consumption_block = ConsumptionBlock(llm, environment, agent_memory)
        self.none_block = EconomyNoneBlock(llm, agent_memory)
        self.trigger_time = 0
        self.token_consumption = 0
        self.dispatcher = BlockDispatcher(llm, agent_memory)
        self.dispatcher.register_blocks(
            [self.work_block, self.consumption_block, self.none_block]
        )

    async def forward(self, step, plan_context):
        """Coordinate economic activity execution.

        Workflow:
            1. Use dispatcher to select appropriate handler
            2. Delegate execution to selected block
        """
        self.trigger_time += 1
        dispatch_context = DotDict({"current_intention": step["intention"]})
        selected_block = await self.dispatcher.dispatch(dispatch_context)
        
        if selected_block is None:
            node_id = await self.memory.stream.add_economy(description=f"I finished {step['intention']}")
            return {
                "success": True,
                "evaluation": f'Finished {step["intention"]}',
                "consumed_time": random.randint(1, 120),
                "node_id": node_id,
            }
        result = await selected_block.forward(step, plan_context)
        return result