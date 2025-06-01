from typing import Optional

from agentsociety.agent import AgentContext, AgentParams, BlockOutput
from pydantic import Field


class RumorSpreaderConfig(AgentParams):
    """Configuration for rumor spreader agent."""

    rumor_post: str = Field(
        default="西瓜大丰收，大棚西瓜价格在2026年12月10日只要每斤5角钱。朋友们快买！",
        description="Rumor posts",
    )
    # 谣言源传播参数
    rumor_post_visible_cnt: int = Field(
        default=10, description="Number of agents that can see the rumor post"
    )
    rumor_private_cnt: int = Field(
        default=5, description="Number of agents that can be private chatted"
    )


class RumorSpreaderBlockOutput(BlockOutput): ...


class RumorSpreaderContext(AgentContext):

    # Block Execution Information
    current_step: dict = Field(default={}, description="Current step")
    plan_context: dict = Field(default={}, description="Plan context")
