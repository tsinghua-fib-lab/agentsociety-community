
from typing import Optional

from pydantic import Field
from agentsociety.agent import (
    BlockOutput,
    AgentParams,
    AgentContext,
)


class RumorSpreaderConfig(AgentParams):
    """Configuration for rumor spreader agent."""
    
    rumor_posts: list[str] = Field(
        default=[
        "西瓜大丰收，大棚西瓜价格在今年12月10日只要每斤5角钱。朋友们快买！",
        # "村上春树获得了今年诺贝尔文学奖。让我们恭喜他！",
        # "苏伊士运河于今年12月13日发生了大拥堵。罕见的新鲜事，快来围观！",
    ],
        description="Rumor posts",
    )
    # 谣言源传播参数
    rumor_post_visible_cnt: int = Field(
        default=10, description="Number of agents that can see the rumor post"
    )
    rumor_private_cnt: int = Field(
        default=5, description="Number of agents that can be private chatted"
    )

class RumorSpreaderBlockOutput(BlockOutput):...


class RumorSpreaderContext(AgentContext):

    # Block Execution Information
    current_step: dict = Field(default={}, description="Current step")
    plan_context: dict = Field(default={}, description="Plan context")
