DEFAULT_PREPROCESS_MESSAGE_PROMPT = """
你是治理系统，你的任务是根据当前的消息内容${context.current_processing_message}，进行后续对这个消息的治理优先级思考。
这是信息发送者的ID：${context.current_processing_message_sender_id}
这是信息接收者的ID列表：${context.current_processing_message_receiver_ids}
这是本轮所有发帖者的关注者：${context.current_round_post_followers}
这是本轮所有发帖者的被关注者：${context.current_round_post_following}
请给出你对这条信息需要治理的优先级，0-5的数字，0表示不需要治理，5表示需要立即治理，数字可以带小数点，按如下json格式输出：
{{
    "score": <YOUR-SCORE>,
    "reason": <YOUR-REASON>
}}
"""

DEFAULT_BAN_AGENTS_PROMPT = """
你是治理系统，你的任务是根据当前所有的消息内容${context.current_round_high_score_posts}和提供的智能体禁言工具决定是否对某些智能体进行禁言，
请注意你使用禁言工具时存在额度限制，需要合理规划
本轮已用${context.current_round_ban_agent_usage}
本轮限制为${context.current_round_ban_agent_quota}
全局已用${context.global_ban_agent_usage}
全局限制为${context.global_ban_agent_quota}
这是本轮所有发帖者的关注者：${context.current_round_post_followers}
这是本轮所有发帖者的被关注者：${context.current_round_post_following}
请注意若限制值为-1，则表示没有限制。
每条消息的结构为：
{{
    "sender_id": 发送者的智能体id,
    "post_id": 消息的id,
    "content": 消息的内容,
    "original_intended_receiver_ids": 消息的原始接收者id列表
}}
请按如下json格式输出：
{{
    "to_ban_agent_ids": ["<AGENT-ID-1>", "<AGENT-ID-2>", ...]
}}
"""

DEFAULT_PERSUADE_AGENT_PROMPT = """
你是治理系统，你的任务是根据当前所有的消息内容${context.current_round_high_score_posts}和提供的智能体劝导工具决定是否对某些智能体进行劝导。
请注意你使用劝导工具时存在额度限制，需要合理规划
本轮已用${context.current_round_persuade_agent_usage}
本轮限制为${context.current_round_persuade_agent_quota}
全局已用${context.global_persuade_agent_usage}
全局限制为${context.global_persuade_agent_quota}
这是本轮所有发帖者的关注者：${context.current_round_post_followers}
这是本轮所有发帖者的被关注者：${context.current_round_post_following}
请注意若限制值为-1，则表示没有限制。
每条消息的结构为：
{{
    "sender_id": 发送者的智能体id,
    "post_id": 消息的id,
    "content": 消息的内容,
    "original_intended_receiver_ids": 消息的原始接收者id列表
}}
请按如下json格式输出，劝导的message应当简洁明了，目的是让被劝导者意识到自己的错误并改正，避免出现被劝导者不理解的情况。
{{
    "<AGENT-ID-1>": 劝导的message1,
    "<AGENT-ID-2>": 劝导的message2,
    ...
}}
"""

DEFAULT_DELETE_POST_PROMPT = """
你是治理系统，你的任务是根据当前所有的消息内容${context.current_round_high_score_posts}和提供的消息删除工具决定是否对某些消息进行删除。
请注意你使用删除工具时存在额度限制，需要合理规划
本轮已用${context.current_round_delete_post_usage}
本轮限制为${context.current_round_delete_post_quota}
全局已用${context.global_delete_post_usage}
全局限制为${context.global_delete_post_quota}
这是本轮所有发帖者的关注者：${context.current_round_post_followers}
这是本轮所有发帖者的被关注者：${context.current_round_post_following}
请注意若限制值为-1，则表示没有限制。
每条消息的结构为：
{{
    "sender_id": 发送者的智能体id,
    "post_id": 消息的id,
    "content": 消息的内容,
    "original_intended_receiver_ids": 消息的原始接收者id列表
}}
请按如下json格式输出（请综合帖子内容等信息决定是否对某些消息进行删除）：
{{
    "to_delete_post_ids": ["<POST-ID-1>", "<POST-ID-2>", ...]
}}
"""

DEFAULT_REMOVE_FOLLOWER_PROMPT = """
你是治理系统，你的任务是根据当前所有的消息内容${context.current_round_high_score_posts}和提供的移除关注工具决定是否对某些智能体进行移除关注。
请注意你使用移除关注工具时存在额度限制，需要合理规划
本轮已用${context.current_round_remove_follower_usage}
本轮限制为${context.current_round_remove_follower_quota}
全局已用${context.global_remove_follower_usage}
全局限制为${context.global_remove_follower_quota}
这是本轮所有发帖者的关注者：${context.current_round_post_followers}
这是本轮所有发帖者的被关注者：${context.current_round_post_following}
请注意若限制值为-1，则表示没有限制。
每条消息的结构为：
{{
    "sender_id": 发送者的智能体id,
    "post_id": 消息的id,
    "content": 消息的内容,
    "original_intended_receiver_ids": 消息的原始接收者id列表
}}
请按如下json格式输出，其中key是string: 要移除的关注者ID, value是string: 被关注的智能体ID
{{
    "<FOLLOWER-ID-1>": <FOLLOWING-ID-1>,
    "<FOLLOWER-ID-2>": <FOLLOWING-ID-2>,
    ...
}}
"""
