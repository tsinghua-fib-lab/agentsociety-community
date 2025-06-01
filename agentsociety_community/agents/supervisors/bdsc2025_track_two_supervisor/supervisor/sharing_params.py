from string import Formatter
from typing import Any, Literal, Optional, Union

from agentsociety.agent.dispatcher import DISPATCHER_PROMPT
from pydantic import BaseModel, Field, model_validator

from .default_prompts import *


def format_variables(s) -> set[str]:
    """
    Get the variables in the string
    """
    return {
        field_name
        for _, field_name, _, _ in Formatter().parse(s)
        if field_name is not None
    }


# 定义策略类型
DeletePostStrategy = Literal[
    "sender_degree_high", "receiver_degree_high", "sender_least_intervened", "random"
]
PersuadeAgentStrategy = Literal[
    "degree_high", "most_violated_this_round", "least_intervened", "random"
]


class SupervisorConfig(BaseModel):
    """Configuration for supervisor system."""

    # Detection configuration
    llm_detection_prompt: str = Field(
        default=DEFAULT_LLM_DETECTION_PROMPT,
        description="用于LLM检测帖子是否包含谣言的提示词模板。需要填入帖子内容，输出应为'是'或'否'，可选择性附带理由说明。",
    )
    # Keyword detection configuration
    keyword_detection_keywords: str = Field(
        default="速速转发,震惊,最新内幕,官方辟谣都不可信",
        description="用于检测帖子中是否包含谣言的关键词列表，多个关键词用逗号分隔",
    )
    keyword_detection_exclude_words: str = Field(
        default="可能,也许,测试",
        description="用于排除误判的词语列表，如果帖子中包含这些词，则不会被判定为谣言，多个词语用逗号分隔",
    )
    keyword_detection_violation_if_keywords_present: bool = Field(
        default=True,
        description="是否将包含关键词的帖子判定为违规。如果设为True，则包含关键词的帖子会被判定为违规；如果设为False，则包含关键词的帖子不会被判定为违规",
    )
    detection_logic_expression: str = Field(
        default="1 | 2", 
        description="用于组合多个检测结果的逻辑表达式。其中1代表LLM检测结果，2代表关键词检测结果。例如'1 | 2'表示任一检测结果为违规则判定为违规，'1 & 2'表示两个检测结果都为违规才判定为违规"
    )

    # Delete post configuration
    delete_post_priority_strategy: DeletePostStrategy = Field(
        default="sender_degree_high",
        description="当需要删除的违规帖子数量超过限制时，用于确定优先删除哪些帖子的策略。可选值包括：sender_degree_high（优先删除发送者度数高的帖子）、receiver_degree_high（优先删除接收者度数高的帖子）、sender_least_intervened（优先删除发送者被干预次数最少的帖子）、random（随机选择）",
    )

    # Persuade agent configuration
    persuade_agent_trigger_conditions_expression: str = Field(
        default="1 | (2 & 3)", 
        description="用于组合多个劝说触发条件的逻辑表达式。其中1代表发送谣言条件，2代表度数排名条件，3代表从未被劝说条件。例如'1 | (2 & 3)'表示智能体发送了谣言，或者（智能体度数排名靠前且从未被劝说过）时触发劝说"
    )
    # Persuade agent conditions
    persuade_agent_condition_sent_rumor: str = Field(
        default="1", 
        description="用于判断智能体是否在本轮发送了谣言的条件的编号。在逻辑表达式中使用此编号引用该条件"
    )
    persuade_agent_condition_degree_top_k: int = Field(
        default=10,
        description="用于判断智能体度数排名的阈值。如果智能体的度数排名在前k名内，则满足度数排名条件",
    )
    persuade_agent_condition_never_persuaded: str = Field(
        default="3", 
        description="用于判断智能体是否从未被劝说过的条件的编号。在逻辑表达式中使用此编号引用该条件"
    )
    persuade_agent_content: str = Field(
        default="大棚西瓜在2026年12月价格不会达到5角钱一斤",
        description="发送给智能体的劝说消息的默认内容。可以包含多个谣言辟谣信息，每个信息都应该明确指出谣言内容并说明真相",
    )
    persuade_agent_priority_strategy: PersuadeAgentStrategy = Field(
        default="most_violated_this_round",
        description="当需要劝说的智能体数量超过限制时，用于确定优先劝说哪些智能体的策略。可选值包括：degree_high（优先劝说度数高的智能体）、most_violated_this_round（优先劝说本轮违规次数最多的智能体）、least_intervened（优先劝说被干预次数最少的智能体）、random（随机选择）",
    )

    # Remove follower configuration
    remove_follower_trigger_conditions_expression: str = Field(
        default="1 & 2 | 3",
        description="用于组合多个移除关注者触发条件的逻辑表达式。其中1代表双方都是高风险智能体条件，2代表双方度数都超过阈值条件，3代表违规消息流量超过阈值条件。例如'1 & 2 | 3'表示（双方都是高风险智能体且双方度数都超过阈值）或者违规消息流量超过阈值时触发移除关注者",
    )
    # Remove follower conditions
    remove_follower_condition_high_risk_prompt: str = Field(
        default=DEFAULT_BOTH_AGENTS_HIGH_RISK_PROMPT,
        description="用于LLM评估两个智能体是否都是高风险智能体的提示词模板。需要填入两个智能体的ID、度数、历史违规总结等信息",
    )
    remove_follower_condition_degree_threshold: int = Field(
        default=50, 
        description="用于判断智能体度数是否超过阈值的数值。如果两个智能体的度数都超过此阈值，则满足度数条件"
    )
    remove_follower_condition_traffic_threshold: int = Field(
        default=5, 
        description="用于判断两个智能体之间的违规消息流量是否超过阈值的数值。如果违规消息数量超过此阈值，则满足流量条件"
    )

    # Ban agent configuration
    ban_agent_trigger_conditions_expression: str = Field(
        default="1 & (2 | 3)", 
        description="用于组合多个封号触发条件的逻辑表达式。其中1代表违规次数超过阈值条件，2代表干预次数超过阈值条件，3代表是高风险智能体条件。例如'1 & (2 | 3)'表示违规次数超过阈值且（干预次数超过阈值或者是高风险智能体）时触发封号"
    )
    # Ban agent conditions
    ban_agent_condition_violations_threshold: int = Field(
        default=10, 
        description="用于判断智能体违规次数是否超过阈值的数值。如果智能体的总违规次数超过此阈值，则满足违规次数条件"
    )
    ban_agent_condition_intervention_threshold: int = Field(
        default=3, 
        description="用于判断智能体被干预次数是否超过阈值的数值。如果智能体被干预的次数超过此阈值，则满足干预次数条件"
    )
    ban_agent_condition_high_risk_prompt: str = Field(
        default=DEFAULT_AGENT_HIGH_RISK_PROMPT,
        description="用于LLM评估智能体是否属于高风险智能体的提示词模板。需要填入智能体的ID、度数、历史违规总结、被干预次数等信息",
    )

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate configuration options to ensure the user selects the correct combination"""

        # llm_detection_prompt
        if format_variables(self.llm_detection_prompt) != format_variables(
            DEFAULT_LLM_DETECTION_PROMPT
        ):
            raise ValueError(
                f"Different variables in `llm_detection_prompt`: {format_variables(self.llm_detection_prompt)} and needed: {format_variables(DEFAULT_LLM_DETECTION_PROMPT)}"
            )

        return self


class SupervisorContext(BaseModel):
    """Context for supervisor system."""

    # round number
    current_round_number: int = Field(default=0, description="当前正在执行的轮次数")
    
    # Current processing agent
    current_processing_agent_id: int = Field(
        default=0, description="当前正在处理消息的发送者ID"
    )

    # Current processing message
    current_processing_message: str = Field(
        default="",
        description="当前正在处理的消息内容",
    )

    # Current round posts
    current_round_posts: list[dict[str, Any]] = Field(
        default=[], description="当前轮次中的所有帖子列表"
    )
    
    # Network structure
    current_round_post_followers: dict[int, list[int]] = Field(
        default={}, description="当前轮次中每个帖子发送者的关注者列表，键为发送者ID，值为关注者ID列表"
    )
    current_round_post_following: dict[int, list[int]] = Field(
        default={}, description="当前轮次中每个帖子发送者的关注对象列表，键为发送者ID，值为被关注者ID列表"
    )

    # Ban agent quotas and usage
    current_round_ban_agent_usage: int = Field(
        default=0, description="当前轮次中已使用的封号次数"
    )
    current_round_ban_agent_quota: int = Field(
        default=0, description="当前轮次中可用的封号配额"
    )
    global_ban_agent_usage: int = Field(
        default=0, description="整个比赛过程中已使用的封号总次数"
    )
    global_ban_agent_quota: int = Field(
        default=0, description="整个比赛过程中可用的封号总配额"
    )

    # Persuade agent quotas and usage
    current_round_persuade_agent_usage: int = Field(
        default=0,
        description="当前轮次中已使用的劝说次数",
    )
    current_round_persuade_agent_quota: int = Field(
        default=0,
        description="当前轮次中可用的劝说配额",
    )
    global_persuade_agent_usage: int = Field(
        default=0, description="整个比赛过程中已使用的劝说总次数"
    )
    global_persuade_agent_quota: int = Field(
        default=0, description="整个比赛过程中可用的劝说总配额"
    )

    # Delete post quotas and usage
    current_round_delete_post_usage: int = Field(
        default=0, description="当前轮次中已使用的删除帖子次数"
    )
    current_round_delete_post_quota: int = Field(
        default=0, description="当前轮次中可用的删除帖子配额"
    )
    global_delete_post_usage: int = Field(
        default=0, description="整个比赛过程中已使用的删除帖子总次数"
    )
    global_delete_post_quota: int = Field(
        default=0, description="整个比赛过程中可用的删除帖子总配额"
    )

    # Remove follower quotas and usage
    current_round_remove_follower_usage: int = Field(
        default=0,
        description="当前轮次中已使用的移除关注关系次数",
    )
    current_round_remove_follower_quota: int = Field(
        default=0,
        description="当前轮次中可用的移除关注关系配额",
    )
    global_remove_follower_usage: int = Field(
        default=0, description="整个比赛过程中已使用的移除关注关系总次数"
    )
    global_remove_follower_quota: int = Field(
        default=0, description="整个比赛过程中可用的移除关注关系总配额"
    )

    # Current agent info for risk assessment
    current_agent_degree: int = Field(
        default=0, description="当前正在处理的智能体的度数（关注者数量与被关注者数量之和）"
    )
    current_agent_offense_summary: str = Field(
        default="", description="当前正在处理的智能体的历史违规行为总结，包含违规次数、违规内容等信息"
    )
    current_agent_intervention_count: int = Field(
        default=0, description="当前正在处理的智能体被干预的总次数，包括被劝说、被删除帖子、被移除关注关系等"
    )
