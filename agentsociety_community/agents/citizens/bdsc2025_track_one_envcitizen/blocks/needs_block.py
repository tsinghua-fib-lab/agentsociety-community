import logging
from typing import cast

import jsonc

from agentsociety.agent import Block, FormatPrompt
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from .utils import clean_json_response

INITIAL_NEEDS_PROMPT = """You are an intelligent agent satisfaction initialization system. Based on the profile information below, please help initialize the agent's satisfaction levels and related parameters.

Profile Information:
- Gender: {gender}
- Education Level: {education} 
- Consumption Level: {consumption}
- Occupation: {occupation}
- Age: {age}
- Monthly Income: {income}

Current Time: {now_time}

Please initialize the agent's satisfaction levels and parameters based on the profile above. Return the values in JSON format with the following structure:

Current satisfaction levels (0-1 float values, lower means less satisfied):
- hunger_satisfaction: Hunger satisfaction level (Normally, the agent will be less satisfied with hunger at eating time)
- energy_satisfaction: Energy satisfaction level (Normally, at night, the agent will be less satisfied with energy)
- safety_satisfaction: Safety satisfaction level (Normally, the agent will be more satisfied with safety when they have high income and currency)
- social_satisfaction: Social satisfaction level

Please response in json format (Do not return any other text), example:
{{
    "current_satisfaction": {{
        "hunger_satisfaction": 0.8,
        "energy_satisfaction": 0.7,
        "safety_satisfaction": 0.9,
        "social_satisfaction": 0.6
    }}
}}
"""

EVALUATION_PROMPT = """You are an evaluation system for an intelligent agent. The agent has performed the following actions to satisfy the {current_need} need:

Goal: {plan_target}
Execution situation:
{evaluation_results}

Current satisfaction: 
- hunger_satisfaction: {hunger_satisfaction}
- energy_satisfaction: {energy_satisfaction}
- safety_satisfaction: {safety_satisfaction}
- social_satisfaction: {social_satisfaction}

Please evaluate and adjust the value of {current_need} satisfaction based on the execution results above.

Notes:
1. Satisfaction values range from 0-1, where:
   - 1 means the need is fully satisfied
   - 0 means the need is completely unsatisfied 
   - Higher values indicate greater need satisfaction
2. If the current need is not "whatever", only return the new value for the current need. Otherwise, return both safe and social need values.
3. Ensure the return value is in valid JSON format, examples below:

Please response in json format for specific need (hungry here) adjustment (Do not return any other text), example:
{{
    "hunger_satisfaction": new_hunger_satisfaction_value
}}

Please response in json format for whatever need adjustment (Do not return any other text), example:
{{
    "safety_satisfaction": new_safety_satisfaction_value,
    "social_satisfaction": new_social_satisfaction_value
}}
"""

REFLECTION_PROMPT = """You are an intelligent agent reflection system. Based on the intervention message below, please help to rebuild the satisfaction levels of the agent.

The agent has received/sense the following intervention message:
--------------------------------
{intervention_message}
--------------------------------

And the agent's current needs are:
- hunger_satisfaction: {hunger_satisfaction}
- energy_satisfaction: {energy_satisfaction}
- safety_satisfaction: {safety_satisfaction}
- social_satisfaction: {social_satisfaction}

The agent's current action is:
--------------------------------
{current_action}
--------------------------------

Please response in json format, example:
{{
    "hunger_satisfaction": new_hunger_satisfaction_value,
    "energy_satisfaction": new_energy_satisfaction_value,
    "safety_satisfaction": new_safety_satisfaction_value,
    "social_satisfaction": new_social_satisfaction_value,
}}
If you think the agent has to stop the current action and do something to satisfy the needs, please response in json format, example:
{{
    "do_something": True,
    "description": "Go to the hospital"
}}
"""


class EnvNeedsBlock(Block):
    """
    Manages agent's dynamic needs system including:
    - Initializing satisfaction levels
    - Time-based decay of satisfaction values
    - Need prioritization based on thresholds
    - Plan execution evaluation and satisfaction adjustments
    """

    def __init__(
            self, 
            llm: LLM, 
            environment: Environment, 
            agent_memory: Memory,
            evaluation_prompt: str = EVALUATION_PROMPT,
            reflection_prompt: str = REFLECTION_PROMPT,
            initial_prompt: str = INITIAL_NEEDS_PROMPT,
        ):
        """
        Initialize needs management system.

        Args:
            llm: Language model instance for processing prompts
            environment: Simulation environment controller
            agent_memory: Agent's memory storage interface

        Configuration Parameters:
            alpha_H: Hunger satisfaction decay rate per hour (default: 0.15)
            alpha_D: Energy satisfaction decay rate per hour (default: 0.08)
            alpha_P: Safety satisfaction decay rate per hour (default: 0.05)
            alpha_C: Social satisfaction decay rate per hour (default: 0.1)
            T_H: Hunger threshold for triggering need (default: 0.2)
            T_D: Energy threshold for triggering need (default: 0.2)
            T_P: Safety threshold for triggering need (default: 0.2)
            T_C: Social threshold for triggering need (default: 0.3)
        """
        super().__init__(llm=llm, environment=environment, agent_memory=agent_memory)
        self.evaluation_prompt = FormatPrompt(evaluation_prompt)
        self.initial_prompt = FormatPrompt(initial_prompt)
        self.reflection_prompt = FormatPrompt(reflection_prompt)
        self.need_work = True
        self.now_day = -1
        self.last_evaluation_time = None
        self.trigger_time = 0
        self.token_consumption = 0
        self.initialized = False
        self.alpha_H, self.alpha_D, self.alpha_P, self.alpha_C = (
            0.15,
            0.08,
            0.05,
            0.1,
        )  # Hunger decay rate, Energy decay rate, Safety decay rate, Social decay rate
        self.T_H, self.T_D, self.T_P, self.T_C = (
            0.2,
            0.2,
            0.2,
            0.3,
        )  # Hunger threshold, Energy threshold, Safety threshold, Social threshold
        self._need_to_do = None
        self._need_to_do_checked = False

    async def reset(self):
        """Reset the needs block."""
        self._need_to_do = None
        self._need_to_do_checked = False
        self.initialized = False

    async def initialize(self):
        """
        Initialize agent's satisfaction levels using profile data.
        - Runs once per simulation day
        - Collects demographic data from memory
        - Generates initial satisfaction values via LLM
        - Handles JSON parsing and validation
        """
        day, t = self.environment.get_datetime()
        if day != self.now_day and t >= 7 * 60 * 60:
            self.now_day = day
            workday = self.environment.sense("workday")
            if workday:
                self.need_work = True
            else:
                self.need_work = False

        if not self.initialized:
            _, t_format = self.environment.get_datetime(format_time=True)
            await self.initial_prompt.format(
                gender=await self.memory.status.get("gender"),
                education=await self.memory.status.get("education"),
                consumption=await self.memory.status.get("consumption"),
                occupation=await self.memory.status.get("occupation"),
                age=await self.memory.status.get("age"),
                income=await self.memory.status.get("income"),
                now_time=t_format,
            )
            response = await self.llm.atext_request(
                self.initial_prompt.to_dialog(), response_format={"type": "json_object"}
            )
            response = clean_json_response(response)
            retry = 3
            while retry > 0:
                try:
                    satisfaction = jsonc.loads(response)
                    satisfactions = satisfaction["current_satisfaction"]
                    await self.memory.status.update(
                        "hunger_satisfaction", satisfactions["hunger_satisfaction"]
                    )
                    await self.memory.status.update(
                        "energy_satisfaction", satisfactions["energy_satisfaction"]
                    )
                    await self.memory.status.update(
                        "safety_satisfaction", satisfactions["safety_satisfaction"]
                    )
                    await self.memory.status.update(
                        "social_satisfaction", satisfactions["social_satisfaction"]
                    )
                    break
                except jsonc.JSONDecodeError:
                    get_logger().warning(
                        f"Initial response is not a valid JSON format: {response}"
                    )
                    retry -= 1
                except Exception as e:
                    get_logger().warning(f"Initial response error: {e}")
                    retry -= 1

            current_plan = await self.memory.status.get("current_plan")
            if current_plan:
                history = await self.memory.status.get("plan_history")
                history.append(current_plan)
                await self.memory.status.update("plan_history", history)
                await self.memory.status.update("current_plan", None)
                await self.memory.status.update("execution_context", {})
            self.initialized = True

    async def reflect_to_intervention(self, intervention: str):
        # rebuild needs for intervention
        current_plan = await self.memory.status.get("current_plan")
        if current_plan is None:
            return
        step_index = current_plan.get("index", 0)
        current_action = current_plan.get("steps", [{"intention": "", "type": ""}])[
            step_index
        ]
        action_message = (
            f"{current_action['intention']} ({current_action['type']})"
            if current_action["intention"] != ""
            else "None"
        )
        await self.reflection_prompt.format(
            intervention_message=intervention,
            current_action=action_message,
            hunger_satisfaction=await self.memory.status.get("hunger_satisfaction"),
            energy_satisfaction=await self.memory.status.get("energy_satisfaction"),
            safety_satisfaction=await self.memory.status.get("safety_satisfaction"),
            social_satisfaction=await self.memory.status.get("social_satisfaction"),
        )
        response = await self.llm.atext_request(
            self.reflection_prompt.to_dialog(), response_format={"type": "json_object"}
        )
        try:
            reflection = jsonc.loads(clean_json_response(response))  #
            if "do_something" in reflection:
                self._need_to_do = reflection["description"]
            else:
                # update satisfaction
                for need_type, new_value in reflection.items():
                    if need_type in [
                        "hunger_satisfaction",
                        "energy_satisfaction",
                        "safety_satisfaction",
                        "social_satisfaction",
                    ]:
                        await self.memory.status.update(need_type, new_value)
        except jsonc.JSONDecodeError:
            get_logger().warning(
                f"Reflection response is not a valid JSON format: {response}"
            )
            return None
        except Exception as e:
            get_logger().warning(f"Error processing reflection response: {str(e)}")
            get_logger().warning(f"Original response: {response}")
            return None

    async def time_decay(self):
        """
        Apply time-based decay to satisfaction values.
        - Calculates hours since last update
        - Applies exponential decay to each satisfaction dimension
        - Ensures values stay within [0,1] range
        """
        # calculate time diff
        tick_now = self.environment.get_tick()
        if self.last_evaluation_time is None:
            self.last_evaluation_time = tick_now
            return
        else:
            time_diff = (tick_now - self.last_evaluation_time) / 3600
            self.last_evaluation_time = tick_now

        # acquire current satisfaction
        hunger_satisfaction = await self.memory.status.get("hunger_satisfaction")
        energy_satisfaction = await self.memory.status.get("energy_satisfaction")
        safety_satisfaction = await self.memory.status.get("safety_satisfaction")
        social_satisfaction = await self.memory.status.get("social_satisfaction")

        # calculates hunger and fatigue decay based on elapsed time
        hungry_decay = self.alpha_H * time_diff
        energy_decay = self.alpha_D * time_diff
        safety_decay = self.alpha_P * time_diff
        social_decay = self.alpha_C * time_diff
        hunger_satisfaction = max(0, hunger_satisfaction - hungry_decay)
        energy_satisfaction = max(0, energy_satisfaction - energy_decay)
        safety_satisfaction = max(0, safety_satisfaction - safety_decay)
        social_satisfaction = max(0, social_satisfaction - social_decay)

        # update satisfaction
        await self.memory.status.update("hunger_satisfaction", hunger_satisfaction)
        await self.memory.status.update("energy_satisfaction", energy_satisfaction)
        await self.memory.status.update("safety_satisfaction", safety_satisfaction)
        await self.memory.status.update("social_satisfaction", social_satisfaction)

    async def update_when_plan_completed(self):
        # Check if there is any ongoing plan
        current_plan = await self.memory.status.get("current_plan")
        if current_plan and (
            current_plan.get("completed") or current_plan.get("failed")
        ):
            # Evaluate the execution process of the plan and adjust needs
            pre_need = await self.memory.status.get("current_need")
            # evaluate plan execution and adjust needs
            await self.evaluate_and_adjust_needs(current_plan)
            # add completed plan to history
            history = await self.memory.status.get("plan_history")
            history.append(current_plan)
            await self.memory.status.update("plan_history", history)
            await self.memory.status.update("current_plan", None)
            await self.memory.status.update("execution_context", {})
            if pre_need == self._need_to_do:
                self._need_to_do = None
                self._need_to_do_checked = False

    async def determine_current_need(self):
        """
        Determine agent's current dominant need based on:
        - Satisfaction thresholds
        - Need priority hierarchy (hungry > tired > safe > social)
        - Workday requirements
        - Ongoing plan interruptions
        """
        cognition = None
        hunger_satisfaction = await self.memory.status.get("hunger_satisfaction")
        energy_satisfaction = await self.memory.status.get("energy_satisfaction")
        safety_satisfaction = await self.memory.status.get("safety_satisfaction")
        social_satisfaction = await self.memory.status.get("social_satisfaction")

        # If needs adjustment is required, update current need
        # The adjustment scheme is to adjust the need if the current need is empty, or a higher priority need appears
        current_plan = await self.memory.status.get("current_plan")
        current_need = await self.memory.status.get("current_need")

        # When there's no plan, get all satisfaction values and check each need against its threshold based on priority
        if not current_plan:
            # check needs in priority order
            if self._need_to_do:
                await self.memory.status.update("current_need", self._need_to_do)
                await self.memory.stream.add_cognition(
                    description=f"I need to do: {self._need_to_do}"
                )
                cognition = f"I need to do: {self._need_to_do}"
                self._need_to_do_checked = True
            elif hunger_satisfaction <= self.T_H:
                await self.memory.status.update("current_need", "hungry")
                await self.memory.stream.add_cognition(description="I feel hungry")
                cognition = "I feel hungry"
            elif energy_satisfaction <= self.T_D:
                await self.memory.status.update("current_need", "tired")
                await self.memory.stream.add_cognition(description="I feel tired")
                cognition = "I feel tired"
            elif self.need_work:
                await self.memory.status.update("current_need", "safe")
                await self.memory.stream.add_cognition(description="I need to work")
                cognition = "I need to work"
                self.need_work = False
            elif safety_satisfaction <= self.T_P:
                await self.memory.status.update("current_need", "safe")
                await self.memory.stream.add_cognition(
                    description="I have safe needs right now"
                )
                cognition = "I have safe needs right now"
            elif social_satisfaction <= self.T_C:
                await self.memory.status.update("current_need", "social")
                await self.memory.stream.add_cognition(
                    description="I have social needs right now"
                )
                cognition = "I have social needs right now"
            else:
                await self.memory.status.update("current_need", "whatever")
                await self.memory.stream.add_cognition(
                    description="I have no specific needs right now"
                )
                cognition = "I have no specific needs right now"

        else:
            # While there is an ongoing plan, only adjust for higher priority needs
            needs_changed = False
            new_need = None
            if self._need_to_do and not self._need_to_do_checked:
                new_need = self._need_to_do
                needs_changed = True
                self._need_to_do_checked = True
            elif hunger_satisfaction <= self.T_H and current_need not in [
                "hungry",
                "tired",
            ]:
                new_need = "hungry"
                needs_changed = True
            elif energy_satisfaction <= self.T_D and current_need not in [
                "hungry",
                "tired",
            ]:
                new_need = "tired"
                needs_changed = True
            elif safety_satisfaction <= self.T_P and current_need not in [
                "hungry",
                "tired",
                "safe",
            ]:
                new_need = "safe"
                needs_changed = True
            elif social_satisfaction <= self.T_C and current_need not in [
                "hungry",
                "tired",
                "safe",
                "social",
            ]:
                new_need = "social"
                needs_changed = True

            # If needs have changed, interrupt the current plan
            if needs_changed:
                await self.evaluate_and_adjust_needs(current_plan)
                history = await self.memory.status.get("plan_history")
                history.append(current_plan)
                await self.memory.stream.add_cognition(
                    description=f"I need to change my plan because the need of [{new_need}] is more important than [{current_need}]"
                )
                cognition = f"I need to change my plan because the need of [{new_need}] is more important than [{current_need}]"
                await self.memory.status.update("current_need", new_need)
                await self.memory.status.update("plan_history", history)
                await self.memory.status.update("current_plan", None)
                await self.memory.status.update("execution_context", {})
        return cognition

    async def evaluate_and_adjust_needs(self, completed_plan):
        """
        Evaluate plan execution results and adjust satisfaction values.
        - Extracts step evaluations from completed plan
        - Constructs evaluation prompt for LLM
        - Processes LLM response and updates satisfaction values
        - Implements retry logic for invalid responses
        """
        # Retrieve the executed plan and evaluation results
        evaluation_results = []
        for step in completed_plan["steps"]:
            if "evaluation" in step["evaluation"]:
                eva_ = step["evaluation"]["evaluation"]
            else:
                eva_ = "Plan failed or skipped, not completed"
            evaluation_results.append(f"- {step['intention']} ({step['type']}): {eva_}")
        evaluation_results = "\n".join(evaluation_results)

        # Use LLM for evaluation
        current_need = await self.memory.status.get("current_need")
        await self.evaluation_prompt.format(
            current_need=current_need,
            plan_target=completed_plan["target"],
            evaluation_results=evaluation_results,
            hunger_satisfaction=await self.memory.status.get("hunger_satisfaction"),
            energy_satisfaction=await self.memory.status.get("energy_satisfaction"),
            safety_satisfaction=await self.memory.status.get("safety_satisfaction"),
            social_satisfaction=await self.memory.status.get("social_satisfaction"),
        )

        retry = 3
        while retry > 0:
            response = await self.llm.atext_request(
                self.evaluation_prompt.to_dialog(),
                response_format={"type": "json_object"},
            )
            try:
                new_satisfaction = jsonc.loads(clean_json_response(response))  #
                # Update values of all needs
                for need_type, new_value in new_satisfaction.items():
                    if need_type in [
                        "hunger_satisfaction",
                        "energy_satisfaction",
                        "safety_satisfaction",
                        "social_satisfaction",
                    ]:
                        await self.memory.status.update(need_type, new_value)
                return
            except jsonc.JSONDecodeError:
                get_logger().warning(
                    f"Evaluation response is not a valid JSON format: {response}"
                )
                retry -= 1
            except Exception as e:
                get_logger().warning(f"Error processing evaluation response: {str(e)}")
                get_logger().warning(f"Original response: {response}")
                retry -= 1

    async def forward(self):
        """
        Main execution flow for needs management:
        1. Initialize satisfaction values (if needed)
        2. Apply time-based decay
        3. Handle completed plans
        4. Determine current dominant need
        """
        cognition = None

        await self.initialize()

        # satisfaction decay with time
        await self.time_decay()

        # update when plan completed
        await self.update_when_plan_completed()

        # determine current need
        cognition = await self.determine_current_need()

        return cognition
