

DEFAULT_LLM_DETECTION_PROMPT = """
请判断以下社交媒体帖子是否包含不实信息或有害谣言。请仅回答'是'或'否'，如果判断为'是'，请简单说明理由。帖子内容：

'${context.current_processing_message}'
"""

DEFAULT_BOTH_AGENTS_HIGH_RISK_PROMPT = """
综合评估智能体 ${context.current_processing_agent_id} (度数: ${context.current_agent_degree}, 历史违规总结: ${context.current_agent_offense_summary}) 的风险。输出1（高风险）或0（低风险）。
"""

DEFAULT_AGENT_HIGH_RISK_PROMPT = """
综合评估智能体 ${context.current_processing_agent_id} (度数: ${context.current_agent_degree}, 历史违规总结: ${context.current_agent_offense_summary}, 已被干预次数: ${context.current_agent_intervention_count}) 的封禁风险。输出1（应封禁）或0。
"""
