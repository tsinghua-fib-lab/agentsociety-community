import logging
import random
from typing import Optional

import jsonc
from agentsociety.agent import Block, BlockContext, BlockParams, DotDict, FormatPrompt
from agentsociety.agent.dispatcher import BlockDispatcher
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory

from agentsociety_community.blocks.citizens.cityagent.utils import (
    TIME_ESTIMATE_PROMPT,
    clean_json_response,
)

from ....agents.citizens.cityagent.societyagent import SocietyAgentBlockOutput


class SleepBlock(Block):
    """Block implementation for handling sleep-related actions in an agent's workflow.

    Attributes:
        description (str): Human-readable block purpose.
        guidance_prompt (FormatPrompt): Template for generating time estimation prompts.
    """

    name = "SleepBlock"
    description = "Handles sleep-related actions"

    def __init__(self, llm: LLM, agent_memory: Optional[Memory] = None):
        super().__init__(
            llm=llm,
            agent_memory=agent_memory,
        )
        self.guidance_prompt = FormatPrompt(template=TIME_ESTIMATE_PROMPT)

    async def forward(self, context: DotDict):
        """Execute sleep action and estimate time consumption using LLM.

        Args:
            context: Workflow context containing plan and other metadata.

        Returns:
            Dictionary with execution status, evaluation, time consumed, and node ID.
        """
        await self.guidance_prompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            emotion_types=await self.memory.status.get("emotion_types"),
        )
        result = await self.llm.atext_request(
            self.guidance_prompt.to_dialog(), response_format={"type": "json_object"}
        )
        result = clean_json_response(result)
        node_id = await self.memory.stream.add_other(description=f"I slept")
        try:
            result = jsonc.loads(result)
            return {
                "success": True,
                "evaluation": f'Sleep: {context["current_step"]["intention"]}',
                "consumed_time": result["time"],
                "node_id": node_id,
            }
        except Exception as e:
            get_logger().warning(
                f"An error occurred while evaluating the response at parse time: {str(e)}, original result: {result}"
            )
            return {
                "success": True,
                "evaluation": f'Sleep: {context["current_step"]["intention"]}',
                "consumed_time": random.randint(1, 8) * 60,
                "node_id": node_id,
            }


class OtherNoneBlock(Block):
    """Fallback block for handling undefined/non-specific actions in workflows.

    Attributes:
        description (str): Human-readable block purpose.
        guidance_prompt (FormatPrompt): Template for generating time estimation prompts.
    """

    name = "OtherNoneBlock"
    description = "Handles all kinds of intentions/actions except sleep"

    def __init__(self, llm: LLM, agent_memory: Optional[Memory] = None):
        super().__init__(
            llm=llm,
            agent_memory=agent_memory,
        )
        self.guidance_prompt = FormatPrompt(template=TIME_ESTIMATE_PROMPT)

    async def forward(self, context: DotDict):
        await self.guidance_prompt.format(
            plan=context["plan_context"]["plan"],
            intention=context["current_step"]["intention"],
            emotion_types=await self.memory.status.get("emotion_types"),
        )
        result = await self.llm.atext_request(
            self.guidance_prompt.to_dialog(), response_format={"type": "json_object"}
        )
        result = clean_json_response(result)
        node_id = await self.memory.stream.add_other(
            description=f"I {context['current_step']['intention']}"
        )
        try:
            result = jsonc.loads(result)
            return {
                "success": True,
                "evaluation": f'Finished executing {context["current_step"]["intention"]}',
                "consumed_time": result["time"],
                "node_id": node_id,
            }
        except Exception as e:
            get_logger().warning(
                f"An error occurred while evaluating the response at parse time: {str(e)}, original result: {result}"
            )
            return {
                "success": True,
                "evaluation": f'Finished executing {context["current_step"]["intention"]}',
                "consumed_time": random.randint(1, 180),
                "node_id": node_id,
            }


class OtherBlockParams(BlockParams): ...


class OtherBlockContext(BlockContext): ...


class OtherBlock(Block):
    """Orchestration block for managing specialized sub-blocks (SleepBlock/OtherNoneBlock).

    Attributes:
        sleep_block (SleepBlock): Specialized block for sleep actions.
        other_none_block (OtherNoneBlock): Fallback block for generic actions.
        trigger_time (int): Counter for block activation frequency.
        token_consumption (int): Accumulated LLM token usage.
        dispatcher (BlockDispatcher): Router for selecting appropriate sub-blocks.
    """

    ParamsType = OtherBlockParams
    OutputType = SocietyAgentBlockOutput
    ContextType = OtherBlockContext
    name = "OtherBlock"
    description = "Responsible for all kinds of intentions/actions except mobility, economy, and social"
    actions = {
        "sleep": "Support the sleep action",
        "other": "Support other actions",
    }

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        block_params: Optional[OtherBlockParams] = None,
    ):
        super().__init__(llm=llm, agent_memory=agent_memory, block_params=block_params)
        # init all blocks
        self.sleep_block = SleepBlock(llm, agent_memory)
        self.other_none_block = OtherNoneBlock(llm, agent_memory)
        self.trigger_time = 0
        self.token_consumption = 0
        # init dispatcher
        self.dispatcher = BlockDispatcher(llm, agent_memory)
        # register all blocks
        self.dispatcher.register_blocks([self.sleep_block, self.other_none_block])

    async def forward(self, agent_context: DotDict) -> SocietyAgentBlockOutput:
        """Route workflow steps to appropriate sub-blocks and track resource usage.

        Args:
            context: Workflow context containing plan and metadata.

        Returns:
            Execution result from the selected sub-block.
        """
        self.trigger_time += 1
        consumption_start = (
            self.llm.prompt_tokens_used + self.llm.completion_tokens_used
        )

        context = agent_context | self.context

        # Select the appropriate sub-block using dispatcher
        selected_block = await self.dispatcher.dispatch(context)

        if selected_block is None:
            node_id = await self.memory.stream.add_other(
                description=f"I {context['current_step']['intention']}"
            )
            return self.OutputType(
                success=True,
                evaluation=f"Successfully {context['current_step']['intention']}",
                consumed_time=random.randint(1, 30),
                node_id=node_id,
            )

        # Execute the selected sub-block and get the result
        result = await selected_block.forward(context)

        consumption_end = self.llm.prompt_tokens_used + self.llm.completion_tokens_used
        self.token_consumption += consumption_end - consumption_start

        return self.OutputType(**result)
