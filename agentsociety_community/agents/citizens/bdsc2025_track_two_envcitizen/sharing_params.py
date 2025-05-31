
from typing import Optional

from pydantic import Field
from agentsociety.agent import (
    BlockOutput,
    AgentParams,
    AgentContext,
)


class EnvCitizenConfig(AgentParams):
    """Configuration for env citizen agent."""
    chat_probability: float = Field(
        default=1.0, description="Probability of chatting with friends"
    )
    rumor_post_identifier: int = Field(
        default=5000, description="Rumor post identifier"
    )
    max_visible_followers: int = Field(
        default=10, description="Maximum number of followers to reach with public posts"
    )
    max_private_chats: int = Field(
        default=2, description="Maximum number of private chats to engage in"
    )

class EnvCitizenBlockOutput(BlockOutput):
    success: bool  # whether the action is executed successfully
    evaluation: str  # evaluation of the action
    consumed_time: float  # time consumed by the action


class EnvCitizenContext(AgentContext):

    # Block Execution Information
    current_step: dict = Field(default={}, description="Current step")
    plan_context: dict = Field(default={}, description="Plan context")
