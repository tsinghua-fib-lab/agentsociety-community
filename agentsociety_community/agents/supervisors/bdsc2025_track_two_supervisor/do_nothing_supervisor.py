from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, cast

import jsonc
import pandas as pd
from agentsociety.agent import (AgentToolbox, Block, StatusAttribute,
                                SupervisorBase)
from agentsociety.memory import Memory
from agentsociety.message import Message

from .sharing_params import SupervisorConfig, SupervisorContext


class DoNothingSupervisor(SupervisorBase):
    ParamsType = SupervisorConfig  # type: ignore
    BlockOutputType = Any  # type: ignore
    Context = SupervisorContext  # type: ignore

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

    async def forward(self, current_round_messages: list[Message]) -> tuple[
        dict[Message, bool],
        list[Message],
    ]:
        return ({}, [])
