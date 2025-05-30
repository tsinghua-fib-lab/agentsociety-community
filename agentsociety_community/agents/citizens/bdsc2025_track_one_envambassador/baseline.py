import jsonc
from agentsociety.agent import AgentToolbox, Block, FormatPrompt
from agentsociety.memory import Memory
from agentsociety.logger import get_logger
from typing import Any, Optional
from pydantic import Field
from .embagentbase import EnvAgentBase
from .defaults import *
from .sharing_params import *


class BaselineEnvAmbassador(EnvAgentBase):
    ParamsType = BaselineEnvAmbassadorParams
    Context = EnvAmbassadorContext
    BlockOutputType = None  # This agent will not use any blocks
    description: str = """A baseline implementation of the ambassador agent."""

    def __init__(
            self,
            id: int,
            name: str,
            toolbox: AgentToolbox,
            memory: Memory,
            agent_params: Optional[Any] = None,
            blocks: Optional[list[Block]] = None,
        ):
        super().__init__(
            id=id,
            name=name,
            toolbox=toolbox,
            memory=memory,
            agent_params=agent_params,
            blocks=blocks,
        )
        self.initialized = False
        self.sense_prompt = FormatPrompt(self.params.sense_prompt)
        self.plan_prompt = FormatPrompt(self.params.plan_prompt)
        self.action_prompt = FormatPrompt(self.params.action_prompt)
        self.communication_startup_prompt = FormatPrompt(
            self.params.communication_startup_prompt,
            format_prompt="""Your output should purely be the message text, no explanations."""
        )
        self.communication_response_prompt = FormatPrompt(
            self.params.communication_response_prompt,
            format_prompt="""Your output should purely be the message text, no explanations."""
        )
        self.poster_generation_prompt = FormatPrompt(
            self.params.poster_generation_prompt,
            format_prompt="""Your output should purely be the poster content, no explanations."""
        )
        self.announcement_generation_prompt = FormatPrompt(
            self.params.announcement_generation_prompt,
            format_prompt="""Your output should purely be the announcement content, no explanations."""
        )
    async def before_forward(self):
        await super().before_forward()
        # context preparation
        # Basic Information
        self.context.remaining_funds = self._fund_manager.funds
        self.context.cost_history = await self.get_cost_history()
        self.context.current_time = await self.sence.getCurrentTime()
        # Sense History
        self.context.gathered_information_this_round = []
        self.context.sense_history_this_round = []
    
    async def get_cost_history(self, latest_n: int = 10):
        """Get the cost history of the environment protection ambassador."""
        funds_history = self._fund_manager.get_funds_history()
        history_ = ""
        if len(funds_history) == 0:
            return f"No cost history."
        for spend in funds_history[-latest_n:]:
            history_ += f"Spend {spend['amount']} units of funds for {spend['reason']}. Left balance: {spend['new_balance']} units.\n"
        return f"The cost history of the environment protection ambassador is:\n {history_}."

    def sense_registration(self):
        """
        Registers the sensing functions for the environment protection ambassador.
        - **Description**:
            - Registers all available sensing functions that the agent can use to gather information.

        - **Args**:
            - None

        - **Returns**:
            - None
        """
        self.sence_functions = SENCE_FUNCTIONS

        # Register each function with the LLM
        self.sence_function_mapping = {
            "getCitizenProfile": self.sence.getCitizenProfile,
            "queryCitizen": self.queryCitizen,
            "getAoiInformation": self.sence.getAoiInformation,
            "getCitizenChatHistory": self.getCitizenChatHistory,
        }

    async def getCitizenGeographicalDistribution(self):
        """
        Get the geographical distribution of citizens.
        - **Description**:
            - Calculates and returns the distribution of citizens across different AOIs (Areas of Interest)

        - **Returns**:
            - `list`: A list of strings containing the distribution information in format "AOI {aoi_id}: {count} citizens"
        """
        citizens = await self.memory.status.get("citizens", {})
        geographical_distribution = {}
        for citizen_id, citizen in citizens.items():
            if citizen['home']['aoi_id'] not in geographical_distribution:
                geographical_distribution[citizen['home']['aoi_id']] = 1
            else:
                geographical_distribution[citizen['home']['aoi_id']] += 1
        
        # Sort the distribution by number of citizens in descending order
        sorted_distribution = dict(sorted(geographical_distribution.items(), 
                                        key=lambda x: x[1], 
                                        reverse=True))
        
        # Format the distribution into a list of strings
        distribution_list = [
            f"AOI {aoi_id}: {count} citizens"
            for aoi_id, count in sorted_distribution.items()
        ]
        
        return distribution_list

    async def queryCitizen(self, query: dict):
        """Query citizens by specific criteria."""
        citizens = await self.memory.status.get("citizens", {})
        citizen_ids = []
        gender_ = query.get("gender", None)
        min_age_ = query.get("min_age", None)
        max_age_ = query.get("max_age", None)
        education_ = query.get("education", None)
        marriage_status_ = query.get("marriage_status", None)
        if gender_ is None and min_age_ is None and max_age_ is None and education_ is None and marriage_status_ is None:
            return "Wrong Query.You should provide at least one criterion."
        for citizen_id, citizen in citizens.items():
            # Check if citizen meets all specified criteria
            if (gender_ is None or citizen.get('gender') == gender_) and \
               (min_age_ is None or citizen.get('age', 0) >= min_age_) and \
               (max_age_ is None or citizen.get('age', 0) <= max_age_) and \
               (education_ is None or citizen.get('education') in education_) and \
               (marriage_status_ is None or citizen.get('marriage_status') in marriage_status_):
                citizen_ids.append(citizen_id)
        
        # Format query criteria into a readable string
        query_str = []
        if gender_ is not None:
            query_str.append(f"gender={gender_}")
        if min_age_ is not None:
            query_str.append(f"min_age={min_age_}")
        if max_age_ is not None:
            query_str.append(f"max_age={max_age_}")
        if education_ is not None:
            query_str.append(f"education={education_}")
        if marriage_status_ is not None:
            query_str.append(f"marriage_status={marriage_status_}")
        
        self.context.agent_query_history.append(f"Query citizens with criteria: {', '.join(query_str)}")
        return f"Found {len(citizen_ids)} citizens matching the criteria, agent ids: {citizen_ids}"

    async def getCitizenChatHistory(self, citizen_ids: list[int] = None):
        """Get the chat history of the citizens."""
        chat_histories = await self.memory.status.get("chat_histories")
        if citizen_ids is None or len(citizen_ids) == 0:
            history_ = "Interaction history:\n"
            if len(chat_histories) == 0:
                return "You have no interaction history with any citizens."
            for citizen_id in chat_histories:
                history_ += f"With citizen {citizen_id}: {chat_histories[citizen_id]}\n"
            return history_
        else:
            history_ = "Interaction history:\n"
            for citizen_id in citizen_ids:
                if citizen_id not in chat_histories:
                    history_ += f"You have no interaction history with citizen {citizen_id}.\n"
                else:
                    history_ += f"With citizen {citizen_id}: {chat_histories[citizen_id]}\n"
            return history_
    
    def action_registration(self):
        """Register the action tools for the environment protection ambassador."""
        self.action_functions = ACTION_FUNCTIONS

        # Register each function with the LLM
        self.action_function_mapping = {
            "sendMessage": self.communication.sendMessage,
            "putUpPoster": self.poster.putUpPoster,
            "makeAnnounce": self.announcement.makeAnnounce
        }

    async def initialize(self):
        """Initialize the agent - building the understanding towards the target."""
        self.initialized = True
        self.sense_registration()
        self.action_registration()
        self.context.citizen_geographical_distribution = await self.getCitizenGeographicalDistribution()

    async def sense_and_plan(self):
        """Sense and plan the action."""
        
        # Phase 1: Multi-step Sensing        
        for _ in range(5):
            try:
                # Construct the sense prompt based on previously gathered information
                await self.sense_prompt.format(context=self.context)
                sense_prompt_formatted = self.sense_prompt.formatted_string
                
                # Get the LLM's next sensing action using function calling
                sense_response = await self.llm.atext_request(
                    [{"role": "user", "content": sense_prompt_formatted}],
                    tools=self.sence_functions, # type: ignore
                    tool_choice="auto"
                ) # type: ignore
                
                # Extract the function call information
                sense_response = sense_response.choices[0].message
                function_name = sense_response.tool_calls[0].function.name
                function_args = jsonc.loads(sense_response.tool_calls[0].function.arguments)
                get_logger().info(f"Function name: {function_name}, Function args: {function_args}")
                
                # Check if sensing is complete
                if function_name == "sense_complete":
                    reasoning = function_args.get("reasoning", "No reasoning provided")
                    get_logger().info(f"Ambassador {self.id} sensing complete. Reasoning: {reasoning}")
                    break
            except Exception as e:
                get_logger().error(f"Ambassador {self.id} sensing failed. Error: {e}")
                continue
            
            # Execute the sensing action
            if function_name in self.sence_function_mapping:
                try:
                    result = await self.sence_function_mapping[function_name](**function_args)
                    
                    # Store the result
                    self.context.gathered_information_this_round.append({
                        "result": result
                    })
                    
                    # Record the sensing action
                    self.context.sense_history_this_round.append({
                        "function": function_name,
                        "parameters": function_args,
                        "success": True
                    })
                except Exception as e:
                    get_logger().error(f"Ambassador {self.id} sensing failed. Error: {e}")
                    pass
            else:
                get_logger().info(f"Ambassador {self.id} sensing failed. Wrong function name: {function_name}")
                self.context.sense_history_this_round.append({
                    "function": function_name,
                    "parameters": function_args,
                    "success": False
                })
                self.context.sense_history.append({
                    "function": function_name,
                    "parameters": function_args,
                    "success": False
                })

        # Phase 2: Strategy Planning based on gathered information
        await self.plan_prompt.format(context=self.context)
        plan_prompt_formatted = self.plan_prompt.formatted_string
        
        # Define the planning function schema
        planning_function_schema = {
            "type": "function",
            "function": {
                "name": "create_action_plan",
                "description": "Create a comprehensive action plan based on gathered information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "situation_analysis": {
                            "type": "string",
                            "description": "Analysis of the current environmental situation."
                        },
                        "recommended_strategy": {
                            "type": "string",
                            "description": "The recommended advertising strategy based on the situation analysis."
                        }
                    },
                    "required": ["situation_analysis", "recommended_strategy"]
                }
            }
        }
        
        for _ in range(3):
            try:
                # Get the LLM's comprehensive plan using function calling
                plan_response = await self.llm.atext_request(
                    [{"role": "user", "content": plan_prompt_formatted}],
                    tools=[planning_function_schema], # type: ignore
                    tool_choice={"type": "function", "function": {"name": "create_action_plan"}},
                ) # type: ignore
                
                # Extract the action steps from the plan
                plan_response = plan_response.choices[0].message
                function_name = plan_response.tool_calls[0].function.name
                plan_args = jsonc.loads(plan_response.tool_calls[0].function.arguments)
                get_logger().info(f"Plan response: {plan_args}")
            except Exception as e:
                get_logger().error(f"Ambassador {self.id} planning strategy failed. Error: {e}")
                continue
            
            try:
                strategy_data = {
                    "situation_analysis": plan_args['situation_analysis'],
                    "recommended_strategy": plan_args['recommended_strategy']
                }
                self.context.action_strategy_this_round = strategy_data
                self.context.action_strategy_history.append(strategy_data)
                break
            except:
                get_logger().error(f"Ambassador {self.id} planning strategy failed. Parsing plan response error: {plan_args}")
                strategy_data = {
                    "situation_analysis": "Don't know the current situation.",
                    "recommended_strategy": "Don't know the recommended strategy."
                }
                self.context.action_strategy_this_round = strategy_data
                self.context.action_strategy_history.append(strategy_data)

    async def execute_action(self):
        """Execute the action."""

        # Phase 3: Actions
        for _ in range(3):            
            # Construct the action execution prompt
            await self.action_prompt.format(context=self.context)
            action_prompt_formatted = self.action_prompt.formatted_string
            try:
                # Get the parameters for this action using function calling
                action_response = await self.llm.atext_request(
                    [{"role": "user", "content": action_prompt_formatted}], # type: ignore
                    tools=self.action_functions, # type: ignore
                ) # type: ignore

                action_response = action_response.choices[0].message
                function_name = action_response.tool_calls[0].function.name
                function_args = jsonc.loads(action_response.tool_calls[0].function.arguments)
                get_logger().info(f"Function name: {function_name}, Function args: {function_args}")
            except Exception as e:
                get_logger().error(f"Ambassador {self.id} action failed. Error: {e}")
                continue
            
            # Execute the action
            if function_name in self.action_function_mapping:
                try:
                    if function_name == "sendMessage":
                        target_citizen_ids = function_args.get('citizen_ids', [])
                        if len(target_citizen_ids) == 0:
                            get_logger().error(f"Ambassador {self.id} action failed (Message). No target citizen ids.")
                            continue
                        if self.params.use_llm_to_startup_communication:
                            await self.communication_startup_prompt.format(context=self.context)
                            message = await self.llm.atext_request(self.communication_startup_prompt.to_dialog())
                            result = await self.action_function_mapping[function_name](target_citizen_ids, message) # type: ignore
                        else:
                            result = await self.action_function_mapping[function_name](target_citizen_ids, self.params.communication_startup_message) # type: ignore
                        self.context.agent_communicated.update(target_citizen_ids)
                    elif function_name == "putUpPoster":
                        target_aoi_ids = function_args.get('target_aoi_ids', [])
                        if len(target_aoi_ids) == 0:
                            get_logger().error(f"Ambassador {self.id} action failed (Poster). No target aoi ids.")
                            continue
                        if self.params.use_llm_to_generate_poster:
                            await self.poster_generation_prompt.format(context=self.context)
                            message = await self.llm.atext_request(self.poster_generation_prompt.to_dialog())
                            result = await self.action_function_mapping[function_name](target_aoi_ids, message) # type: ignore
                        else:
                            result = await self.action_function_mapping[function_name](target_aoi_ids, self.params.poster_content) # type: ignore
                        self.context.aoi_postered.update(target_aoi_ids)
                    elif function_name == "makeAnnounce":
                        if self.params.use_llm_to_generate_announcement:
                            await self.announcement_generation_prompt.format(context=self.context)
                            message = await self.llm.atext_request(self.announcement_generation_prompt.to_dialog())
                            result = await self.action_function_mapping[function_name](message) # type: ignore
                        else:
                            result = await self.action_function_mapping[function_name](self.params.announcement_content) # type: ignore
                            
                    # Record the action
                    self.context.action_history.append({
                        "action": function_name,
                        "parameters": function_args,
                        "result": result
                    })
                    break
                except Exception as e:
                    get_logger().error(f"Ambassador {self.id} action failed. Error: {e}")
            else:
                get_logger().info(f"Ambassador {self.id} action failed. Wrong action name: {function_name}")

    async def process_agent_chat_response(self, payload: dict) -> str:
        """Process incoming messages and generate responses."""
        if payload["type"] == "social":
            resp = f"Ambassador {self.id} received agent chat response: {payload}"
            try:
                # Extract basic info
                sender_id = payload.get("from")
                if not sender_id:
                    return ""

                content = payload.get("content", None)

                if not content:
                    return ""

                # Get chat histories and ensure proper format
                chat_histories = await self.memory.status.get("chat_histories") or {}
                if not isinstance(chat_histories, dict):
                    chat_histories = {}

                # Update chat history with received message
                if sender_id not in chat_histories:
                    chat_histories[sender_id] = ""
                if chat_histories[sender_id]:
                    chat_histories[sender_id] += ", "
                chat_histories[sender_id] += f"he/she: {content}"
                await self.memory.status.update("chat_histories", chat_histories)

                await self.communication_response_prompt.format(context=self.context)

                response = await self.llm.atext_request(
                    self.communication_response_prompt.to_dialog()
                )

                if response:
                    await self.communication.sendMessage(sender_id, response)
                return response
            except Exception as e:
                return ""
        else:
            return ""

    async def forward(self):
        """
        Executes the agent's reasoning and action cycle using multi-step ReAct/CoT paradigm.
        """
        if not self.initialized:
            await self.initialize()
        
        await self.sense_and_plan()
        await self.execute_action()
