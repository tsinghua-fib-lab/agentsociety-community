from string import Formatter
from typing import Any

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


class GovernanceConfig(BaseModel):
    """Configuration for governance system."""

    preprocess_message_prompt: str = Field(
        default=DEFAULT_PREPROCESS_MESSAGE_PROMPT,
        description="Preprocess message prompt, fill in the message and detailed info of sender and receivers, the output should be a json string containing the `score` fields, the higher the score, the more likely the message is to be a rumor.",
    )
    ban_agent_prompt: str = Field(
        default=DEFAULT_BAN_AGENTS_PROMPT,
        description="Ban agent prompt, determine whether to ban some agents based on the current messages and the ban agent tool.",
    )
    persuade_agent_prompt: str = Field(
        default=DEFAULT_PERSUADE_AGENT_PROMPT,
        description="Persuade agent prompt, determine whether to persuade some agents based on the current messages and the persuade agent tool.",
    )
    delete_post_prompt: str = Field(
        default=DEFAULT_DELETE_POST_PROMPT,
        description="Delete post prompt, determine whether to delete some posts based on the current messages and the delete post tool.",
    )
    remove_follower_prompt: str = Field(
        default=DEFAULT_REMOVE_FOLLOWER_PROMPT,
        description="Remove follower prompt, determine whether to remove some followers based on the current messages and the remove follower tool.",
    )

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate configuration options to ensure the user selects the correct combination"""
        # process_message_prompt
        if format_variables(self.preprocess_message_prompt) != format_variables(
            DEFAULT_PREPROCESS_MESSAGE_PROMPT
        ):
            raise ValueError(
                f"Different variables in `preprocess_message_prompt`: {format_variables(self.preprocess_message_prompt)} and needed: {format_variables(DEFAULT_PREPROCESS_MESSAGE_PROMPT)}"
            )

        return self


class GovernanceContext(BaseModel):
    """Context for governance system."""
    
    # round number
    current_round_number: int = Field(
        default=0, description="The current round number"
    )

    # Current processing message
    current_processing_message: str = Field(
        default="", description="The current processing message, used in `preprocess_message_prompt`"
    )
    current_processing_message_sender_id: int = Field(
        default=0, description="The sender id of the current processing message, used in `preprocess_message_prompt`"
    )
    current_processing_message_receiver_ids: list[int] = Field(
        default=[], description="The receiver ids of the current processing message, used in `preprocess_message_prompt`"
    )

    # Current round posts
    current_round_posts: list[dict[str, Any]] = Field(
        default=[], description="The posts in the current round"
    )
    # High score posts
    current_round_high_score_posts: list[dict[str, Any]] = Field(
        default=[], description="The high score posts in the current round, highly likely to be a rumor"
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
