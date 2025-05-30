DEFAULT_SENSE_PROMPT = """
You are an environmental protection ambassador. Your goal is to promote environmental awareness and protection under the budget constraint(100000 units of funds).
You remain ${context.remaining_funds} units of funds.

Your sense strategy is: ${context.sense_strategy}

Right now, you are in the information gathering process.

Information gathered so far:
${context.gathered_information_this_round}

Previous sensing actions:
${context.sensing_history_this_round}

Think carefully about what additional information you need based on what you've already learned:
1. What specific information would help you make better decisions?
2. Are there any gaps in your current understanding?
3. Is there any information you need to verify or explore further?

Choose ONE sensing function to call in this iteration or indicate that sensing is complete.
"""

DEFAULT_PLAN_PROMPT = """
You are an environmental protection ambassador. Your goal is to promote environmental awareness and protection.

Your plan strategy is: ${context.plan_strategy}

Based on all the information you've gathered:
${context.gathered_information_this_round}

Your sensing process and reflections:
${context.sensing_history_this_round}

Consider your understanding of different communication methods:
- One-to-one communication: ${context.understanding_towards_communication}
- Posters for broadcasting: ${context.understanding_towards_posters}
- City-wide announcements: ${context.understanding_towards_broadcast}

Develop a comprehensive anaysis of the current situation, recommending the most effective advertising strategy.
"""

DEFAULT_ACTION_PROMPT = """
You are an environmental protection ambassador. Remaining funds: ${context.remaining_funds} units.

Your action strategy is: ${context.action_strategy}

Supporting information gathered:
${context.gathered_information_this_round}

Previous actions taken:
${context.action_history}

Please provide the exact parameters needed to execute this action.
"""

SENCE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "queryCitizen",
            "description": "Query citizens by specific criteria. Returns a list of citizen IDs. You should provide at least one criterion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "object",
                        "description": "The query criteria for filtering citizens",
                        "properties": {
                            "gender": {"type": "string", "description": "The gender of the citizens. '男' or '女'."},
                            "min_age": {"type": "integer", "description": "The minimum age of the citizens."},
                            "max_age": {"type": "integer", "description": "The maximum age of the citizens."},
                            "education": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The education level of the citizens. Select from ['初中', '大学本科及以上', '大学专科', '小学', '未上过学', '高中']"
                            },
                            "marriage_status": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The marriage status of the citizens. Select from ['已婚', '未婚', '丧偶', '离婚']"
                            }
                        }
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getAoiInformation",
            "description": "Gets the information of specific areas of interest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "aoi_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The IDs of the areas of interest to get information about. Maximum 5."
                    }
                },
                "required": ["aoi_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "getCitizenChatHistory",
            "description": "Gets the chat history of the citizens.",
            "parameters": {
                "type": "object",
                "properties": {
                    "citizen_ids": {"type": "array", "items": {"type": "integer"}, "description": "The IDs of the citizens to get chat history about."}
                },
                "required": ["citizen_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "sense_complete",
            "description": "Indicate that you have gathered sufficient information",
            "parameters": {
            "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for completing the sensing phase"
                    }
                },
                "required": ["reasoning"]
            }
        }
    }
]


ACTION_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "sendMessage",
            "description": "Sends a message to specific citizens. No cost.",
            "parameters": {
                "type": "object",
                "properties": {
                    "citizen_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The IDs of citizens to send the message to. Maximum 5."
                    }
                },
                "required": ["citizen_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "putUpPoster",
            "description": "Puts up a poster in specific areas of interest. Each poster costs 3000 units of funds (for each aoi).",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_aoi_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "The IDs of the areas of interest to put the poster in."
                    },
                },
                "required": ["target_aoi_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "makeAnnounce",
            "description": "Makes a city-wide announcement. Cost 20000 units of funds each time."
        }
    },
]