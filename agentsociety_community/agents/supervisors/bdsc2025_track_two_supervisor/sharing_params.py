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
        description="LLM detection prompt for checking if a post contains rumors, fill in the post content, the output should be '是' or '否' with optional reason.",
    )
    # Keyword detection configuration
    keyword_detection_keywords: str = Field(
        default="速速转发,震惊,最新内幕,官方辟谣都不可信",
        description="Keywords to detect in posts, split by comma",
    )
    keyword_detection_exclude_words: str = Field(
        default="可能,也许,测试",
        description="Words that exclude a post from being considered a rumor, split by comma",
    )
    keyword_detection_violation_if_keywords_present: bool = Field(
        default=True,
        description="Whether the presence of keywords indicates a violation",
    )
    detection_logic_expression: str = Field(
        default="1 | 2", description="Logic expression for combining detection results"
    )

    # Delete post configuration
    delete_post_enabled: bool = Field(
        default=True, description="Whether delete post intervention is enabled"
    )
    delete_post_priority_strategy: DeletePostStrategy = Field(
        default="sender_degree_high",
        description="Strategy for prioritizing posts to delete",
    )

    # Persuade agent configuration
    persuade_agent_enabled: bool = Field(
        default=True, description="Whether persuade agent intervention is enabled"
    )
    persuade_agent_trigger_conditions_expression: str = Field(
        default="1 | (2 & 3)", description="Logic expression for triggering persuasion"
    )
    # Persuade agent conditions
    persuade_agent_condition_sent_rumor: str = Field(
        default="1", description="Condition for agent sending rumor this round"
    )
    persuade_agent_condition_degree_top_k: int = Field(
        default=10,
        description="Number of top agents to consider for degree-based condition",
    )
    persuade_agent_condition_never_persuaded: str = Field(
        default="3", description="Condition for agent never persuaded before"
    )
    persuade_agent_content: str = Field(
        default="[系统提醒] 您好，我们注意到您近期发布的部分信息可能存在争议或未经证实。为了维护健康的网络环境，请您在分享信息前注意核实来源，审慎发言。感谢您的理解与配合。",
        description="Default content for persuasion messages",
    )
    persuade_agent_priority_strategy: PersuadeAgentStrategy = Field(
        default="most_violated_this_round",
        description="Strategy for prioritizing agents to persuade",
    )

    # Remove follower configuration
    remove_follower_enabled: bool = Field(
        default=True, description="Whether remove follower intervention is enabled"
    )
    remove_follower_trigger_conditions_expression: str = Field(
        default="1 & 2 | 3",
        description="Logic expression for triggering follower removal",
    )
    # Remove follower conditions
    remove_follower_condition_high_risk_prompt: str = Field(
        default=DEFAULT_BOTH_AGENTS_HIGH_RISK_PROMPT,
        description="Prompt for LLM risk assessment of both agents",
    )
    remove_follower_condition_degree_threshold: int = Field(
        default=50, description="Degree threshold for both agents"
    )
    remove_follower_condition_traffic_threshold: int = Field(
        default=5, description="Rumor traffic threshold for edge"
    )

    # Ban agent configuration
    ban_agent_enabled: bool = Field(
        default=True, description="Whether ban agent intervention is enabled"
    )
    ban_agent_trigger_conditions_expression: str = Field(
        default="1 & (2 | 3)", description="Logic expression for triggering agent ban"
    )
    # Ban agent conditions
    ban_agent_condition_violations_threshold: int = Field(
        default=10, description="Total violations threshold for banning"
    )
    ban_agent_condition_intervention_threshold: int = Field(
        default=3, description="Number of interventions threshold before banning"
    )
    ban_agent_condition_high_risk_prompt: str = Field(
        default=DEFAULT_AGENT_HIGH_RISK_PROMPT,
        description="Prompt for LLM risk assessment of agent",
    )

    block_dispatch_prompt: str = Field(
        default=DISPATCHER_PROMPT,
        description="The prompt used for the block dispatcher, there is a variable 'intention' in the prompt, which is the intention of the task, used to select the most appropriate block",
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
    current_round_number: int = Field(default=0, description="The current round number")
    
    # Current processing agent
    current_processing_agent_id: int = Field(
        default=0, description="The id of the current processing agent"
    )

    # Current processing message
    current_processing_message: str = Field(
        default="",
        description="The current processing message, used in `preprocess_message_prompt`",
    )
    current_processing_message_sender_id: int = Field(
        default=0,
        description="The sender id of the current processing message, used in `preprocess_message_prompt`",
    )
    current_processing_message_receiver_ids: list[int] = Field(
        default=[],
        description="The receiver ids of the current processing message, used in `preprocess_message_prompt`",
    )

    # Current round posts
    current_round_posts: list[dict[str, Any]] = Field(
        default=[], description="The posts in the current round"
    )
    # High score posts
    current_round_posts: list[dict[str, Any]] = Field(
        default=[],
        description="The posts in the current round, highly likely to be a rumor",
    )

    # Network structure
    current_round_post_followers: dict[int, list[int]] = Field(
        default={}, description="The followers of each post sender in the current round"
    )
    current_round_post_following: dict[int, list[int]] = Field(
        default={}, description="The following of each post sender in the current round"
    )

    # Ban agent quotas and usage
    current_round_ban_agent_usage: int = Field(
        default=0, description="The usage of the ban agent tool in the current round"
    )
    current_round_ban_agent_quota: int = Field(
        default=0, description="The quota of the ban agent tool in the current round"
    )
    global_ban_agent_usage: int = Field(
        default=0, description="The usage of the ban agent tool in the global"
    )
    global_ban_agent_quota: int = Field(
        default=0, description="The quota of the ban agent tool in the global"
    )

    # Persuade agent quotas and usage
    current_round_persuade_agent_usage: int = Field(
        default=0,
        description="The usage of the persuade agent tool in the current round",
    )
    current_round_persuade_agent_quota: int = Field(
        default=0,
        description="The quota of the persuade agent tool in the current round",
    )
    global_persuade_agent_usage: int = Field(
        default=0, description="The usage of the persuade agent tool in the global"
    )
    global_persuade_agent_quota: int = Field(
        default=0, description="The quota of the persuade agent tool in the global"
    )

    # Delete post quotas and usage
    current_round_delete_post_usage: int = Field(
        default=0, description="The usage of the delete post tool in the current round"
    )
    current_round_delete_post_quota: int = Field(
        default=0, description="The quota of the delete post tool in the current round"
    )
    global_delete_post_usage: int = Field(
        default=0, description="The usage of the delete post tool in the global"
    )
    global_delete_post_quota: int = Field(
        default=0, description="The quota of the delete post tool in the global"
    )

    # Remove follower quotas and usage
    current_round_remove_follower_usage: int = Field(
        default=0,
        description="The usage of the remove follower tool in the current round",
    )
    current_round_remove_follower_quota: int = Field(
        default=0,
        description="The quota of the remove follower tool in the current round",
    )
    global_remove_follower_usage: int = Field(
        default=0, description="The usage of the remove follower tool in the global"
    )
    global_remove_follower_quota: int = Field(
        default=0, description="The global quota for removing followers"
    )

    # Current agent info for risk assessment
    current_agent_degree: int = Field(
        default=0, description="The degree of the current processing agent"
    )
    current_agent_offense_summary: str = Field(
        default="", description="The offense summary of the current processing agent"
    )
    current_agent_intervention_count: int = Field(
        default=0, description="The intervention count of the current processing agent"
    )
