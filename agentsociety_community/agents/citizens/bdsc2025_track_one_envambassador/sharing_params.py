from pydantic import Field
from agentsociety.agent import AgentParams, AgentContext
from .defaults import *


class BaselineEnvAmbassadorParams(AgentParams):
    # Sense
    sense_prompt: str = Field(
        default=DEFAULT_SENSE_PROMPT,
        description="用于感知阶段的提示词。",
    )

    # Plan
    plan_prompt: str = Field(
        default=DEFAULT_PLAN_PROMPT,
        description="用于规划阶段的提示词。",
    )

    # Action
    action_prompt: str = Field(
        default=DEFAULT_ACTION_PROMPT,
        description="用于行动阶段的提示词。",
    )

    # Communication
    use_llm_to_startup_communication: bool = Field(
        default=True,
        description="是否使用LLM来启动与市民的沟通。",
    )
    communication_startup_message: str = Field(
        default="你好，我是环保大使。我们需要你的帮助来保护我们的环境。",
        description="（如果设置use_llm_to_startup_communication为False）这是环保大使首次与市民沟通时发送的消息。",
    )
    communication_startup_prompt: str = Field(
        default="你正在帮助生成环保大使的启动消息。请指出环境保护的重要性。",
        description="（如果设置use_llm_to_startup_communication为True）环保大使首次与市民沟通时的具体沟通策略。",
    )
    communication_response_prompt: str = Field(
        default="你正在帮助生成聊天回复。我是一名环保大使。请帮助我生成对消息的回复。",
        description="环保大使生成对市民发送消息的回复时的具体沟通策略。",
    )

    # Poster
    use_llm_to_generate_poster: bool = Field(
        default=True,
        description="是否使用LLM来生成海报内容。",
    )
    poster_content: str = Field(
        default="保护环境，保护未来。",
        description="（如果设置use_llm_to_generate_poster为False）海报的内容。",
    )
    poster_generation_prompt: str = Field(
        default="你正在帮助生成环保大使的海报内容。请指出环境保护的重要性。",
        description="（如果设置use_llm_to_generate_poster为True）环保大使生成海报内容时的具体策略。",
    )

    # Announcement
    use_llm_to_generate_announcement: bool = Field(
        default=True,
        description="是否使用LLM来生成公告内容。",
    )
    announcement_content: str = Field(
        default="保护环境，保护未来。",
        description="（如果设置use_llm_to_generate_announcement为False）公告的内容。",
    )
    announcement_generation_prompt: str = Field(
        default="你正在帮助生成环保大使的公告内容。请指出环境保护的重要性。",
        description="（如果设置use_llm_to_generate_announcement为True）环保大使生成公告内容时的具体策略。",
    )


class EnvAmbassadorContext(AgentContext):
    # Basic Information
    remaining_funds: int = Field(
        default=100000,
        description="环保大使的剩余资金。",
    )
    citizen_geographical_distribution: list[str] = Field(
        default=[],
        description="该地区市民的分布情况。列表按市民数量降序排列。示例：'AOI x1: x2市民'，x1是aoi_id，x2是该aoi中的市民数量。",
    )
    cost_history: str = Field(
        default="暂无支出记录。",
        description="环保大使产生的支出历史。",
    )
    current_time: str = Field(
        default="不知道当前时间。",
        description="当前时间。",
    )
    

    # Sense History
    gathered_information_this_round: list[dict] = Field(
        default=[],
        description="环保大使在本轮收集的信息。",
    )
    sense_history_this_round: list[dict] = Field(
        default=[],
        description="环保大使在本轮采取的感知行动历史。",
    )
    agent_query_history: list[str] = Field(
        default=[],
        description="环保大使进行的代理查询历史。",
    )
    agent_communicated: set[int] = Field(
        default=set(),
        description="环保大使已沟通过的代理历史。",
    )
    aoi_postered: set[int] = Field(
        default=set(),
        description="环保大使已张贴海报的aoi历史。",
    )

    # Plan
    action_strategy_this_round: dict = Field(
        default={},
        description="用于规划阶段的策略。",
    )
    action_strategy_history: list[dict] = Field(
        default=[],
        description="环保大使采取的行动策略历史。",
    )

    # Action History
    action_history: list[dict] = Field(
        default=[],
        description="环保大使采取的行动历史。",
    )