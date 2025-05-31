from typing import Optional

from pydantic import Field
from agentsociety.agent import (
    BlockOutput,
    AgentParams,
    AgentContext,
)
from .needs_block import INITIAL_NEEDS_PROMPT
from .plan_block import DETAILED_PLAN_PROMPT


class SocietyAgentConfig(AgentParams):
    """Configuration for society agent."""

    enable_cognition: bool = Field(
        default=True, description="Whether to enable cognition"
    )
    UBI: float = Field(default=0, description="Universal Basic Income")
    num_labor_hours: int = Field(
        default=168, description="Number of labor hours per month"
    )
    productivity_per_labor: float = Field(
        default=1, description="Productivity per labor hour"
    )
    time_diff: int = Field(
        default=30 * 24 * 60 * 60, description="Time difference between two triggers"
    )

    # Maxlow's Needs
    need_initialization_prompt: str = Field(
        default=INITIAL_NEEDS_PROMPT, description="Initial needs prompt"
    )

    # Planned-Behavior
    max_plan_steps: int = Field(
        default=6, description="Maximum number of steps in a plan"
    )
    plan_generation_prompt: str = Field(
        default=DETAILED_PLAN_PROMPT, description="Plan generation prompt"
    )


class SocietyAgentBlockOutput(BlockOutput):
    success: bool  # whether the action is executed successfully
    evaluation: str  # evaluation of the action
    consumed_time: int  # time consumed by the action
    node_id: Optional[int]  # stream memory node id


class SocietyAgentContext(AgentContext):
    # Normal Information
    current_time: str = Field(
        default="Don't know", description="Current time, in HH:MM:SS format"
    )
    current_need: str = Field(
        default="Don't know", description="Current need"
    )  # runtime
    current_intention: str = Field(
        default="Don't know", description="Current intention"
    )  # runtime
    current_emotion: str = Field(default="Don't know", description="Current emotion")
    current_thought: str = Field(default="Don't know", description="Current thought")
    current_location: str = Field(default="Don't know", description="Current location")

    # Environment Information
    area_information: str = Field(default="Don't know", description="Area information")
    weather: str = Field(default="Don't know", description="Weather information")
    temperature: str = Field(
        default="Don't know", description="Temperature information"
    )
    other_information: str = Field(
        default="Don't know", description="Other information"
    )

    # Plan Information
    plan_target: str = Field(default="Don't know", description="Plan target")
    max_plan_steps: int = Field(
        default=6, description="Maximum number of steps in a plan"
    )

    # Block Execution Information
    current_step: dict = Field(default={}, description="Current step")
    plan_context: dict = Field(default={}, description="Plan context")
