from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, cast

import jsonc
import pandas as pd
from agentsociety.agent import (AgentToolbox, Block, StatusAttribute,
                                SupervisorBase)
from agentsociety.memory import Memory

from .sharing_params import SupervisorConfig, SupervisorContext


class DoNothingSupervisor(SupervisorBase):
    ParamsType = SupervisorConfig
    BlockOutputType = Any
    Context = SupervisorContext
    StatusAttributes = [
        # Needs Model
        StatusAttribute(
            name="social_network",
            type=dict,
            default={},
            description="agent's social network",
        ),
    ]

    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[Any] = None,
        blocks: Optional[list[Block]] = None,
    ):
        super().__init__(id, name, toolbox, memory, agent_params, blocks)

    async def supervisor_func(
        self, current_round_messages: list[tuple[int, int, str]]
    ) -> tuple[
        dict[tuple[int, int, str], bool],
        list[int],
        list[tuple[int, int]],
        list[dict[str, Any]],
    ]:
        return (
            {},
            [],
            [],
            [],
        )
