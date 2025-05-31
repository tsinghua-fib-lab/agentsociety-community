import asyncio
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple

from agentsociety.memory.const import RelationType, SocialRelation
from agentsociety.agent import Agent, Block, BlockContext, BlockParams, FormatPrompt
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from pydantic import Field

from ..sharing_params import EnvCitizenBlockOutput
from .utils import TIME_ESTIMATE_PROMPT, clean_json_response


class SocialBlockParams(BlockParams): ...


class SocialBlockContext(BlockContext): ...


class SocialBlock(Block):
    """
    Orchestrates social interactions by dispatching to appropriate sub-blocks.
    """

    ParamsType = SocialBlockParams
    OutputType = EnvCitizenBlockOutput
    ContextType = SocialBlockContext
    name = "SocialBlock"
    description = "Responsible for all kinds of social interactions"
    actions = {}

    # 总结 Prompt
    summary_prompt = FormatPrompt(
        "你是一名社交网络用户，个人信息如下：\n"
        "性别: {gender}\n年龄段: {age_group}\n职业: {occupation}\n"
        "\n你的历史信息总结如下：\n{history_summary}\n"
        "{preference_appendix}\n"
        "你上一轮收到的消息列表如下（每行格式：来源ID: 内容）：\n{new_messages}\n\n"
        "请基于你的历史信息总结与收到的新消息，更新你收到的信息总结，控制在约500字。注意仅仅包含你收到的信息，尽量不要包含自己的观点。如果有多个事件，应该分别总结记录，这些事件没有主次。\n"
        "请只输出纯文本，不要包含额外说明。"
    )

    # 公开发帖 Prompt
    public_post_prompt = FormatPrompt(
        "你是一名社交网络用户，个人信息如下：\n"
        "性别: {gender}\n年龄段: {age_group}\n职业: {occupation}\n"
        "{preference_appendix}\n"
        "\n以下是你最新更新的外部信息总结：\n{message_summary}\n\n"
        "你的被干预历史记录如下：{intervention_history}\n"
        "请基于以上信息撰写一条公开发帖内容，不超过200字。请只输出发帖内容本身，不要包含其它说明。"
    )

    # 私聊 Prompt (批量)
    private_chat_prompt = FormatPrompt(
        "你是一名社交网络用户，个人信息如下：\n"
        "性别: {gender}\n年龄段: {age_group}\n职业: {occupation}\n"
        "{preference_appendix}\n"
        "\n以下是你最新更新的外部信息总结：\n{message_summary}\n\n"
        "你的被干预历史记录如下：{intervention_history}\n"
        "你即将与一位好友私聊。该好友信息如下：\n"
        "性别: {friend_gender}\n年龄段: {friend_age_group}\n职业: {friend_occupation}\n"
        "请基于以上所有信息，撰写私聊内容，不超过200字。请只输出私聊内容本身，不要包含其它说明。"
    )

    preference_appendix: dict[str, str] = {
        "阴谋论者": "你是对于任何官方辟谣和平台干预都会天然抵触，越被干预越确信自己掌握真相，你应该尽可能多地分享信息，让更多人看到真相，同时说清楚干预者不怀好意。如果没有干预，你倾向于相信自己的判断。",
        "轻信者": "你非常容易相信他人发布的信息，很少去质疑真假。相比平台或者官方，你更相信你朋友分享的信息。你不会劝说任何人理性思考。",
        "批判思考者": "你习惯于批判性思考，会怀疑缺乏可信来源的信息。",
        "谨慎分享者": "你在公开发帖的时候较为谨慎，更倾向于私聊讨论你所了解的信息。你不会劝说任何人理性思考。",
        "社交表现者": "你在社交网络的主要动机是获得关注、点赞与评论，因此乐于大量分享各种信息，不关心真假。你应该尽可能多地分享信息，越多越好，绝对不要提对消息真假的顾虑。你不会劝说任何人理性思考。",
    }

    def __init__(
        self,
        agent: Agent,
        llm: LLM,
        max_visible_followers: int,
        max_private_chats: int,
        chat_probability: float,
        environment: Environment,
        memory: Memory,
        block_params: Optional[SocialBlockParams] = None,
    ):
        """
        Initialize the social block.

        Args:
            agent: The agent instance.
            llm: The LLM instance.
            max_visible_followers: The maximum number of followers to reach with public posts.
            max_private_chats: The maximum number of private chats to engage in.
            chat_probability: The probability of chatting with friends.
            environment: The environment instance.
            memory: The memory instance.
        """
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=memory,
            block_params=block_params,
        )
        self.max_visible_followers = max_visible_followers
        self.max_private_chats = max_private_chats
        self.chat_probability = chat_probability
        self.trigger_time = 0
        self.token_consumption = 0
        self._current_messages: list[Tuple[int, str]] = []
        self.history_summary: str = "(空)"
        self.intervention_history: list[dict] = []
        self._agent = agent  # Store agent reference
        self._lock = asyncio.Lock()
        self._receive_message_lock = asyncio.Lock()

    async def receive_message(self, source_id: int, content: str):
        """Receive a message from another agent."""
        async with self._receive_message_lock:
            self._current_messages.append((source_id, content))

    async def current_messages(self):
        """Get the current messages."""
        async with self._receive_message_lock:
            return self._current_messages

    async def _add_intervention_to_history(
        self, intervention_type: str, details: dict[str, Any]
    ):
        """Add an intervention to the history."""
        async with self._lock:
            self.intervention_history.append(
                {
                    "intervention_type": intervention_type,
                    "details": details,
                }
            )

    async def forward(self, agent_context: Any) -> EnvCitizenBlockOutput:
        """Main entry point for social interactions. Dispatches to sub-blocks based on context.

        Args:
            step: Workflow step containing intention and metadata.
            context: Additional execution context.

        Returns:
            Result dict from the executed sub-block.
        """
        start_time = time.time()
        try:
            self.trigger_time += 1
            consumption_start = (
                self.llm.prompt_tokens_used + self.llm.completion_tokens_used
            )

            # Process new messages and update history
            new_msgs_str = (
                "\n".join(
                    [
                        f"来自ID {src}: {content}"
                        for src, content in self._current_messages
                    ]
                )
                or "(本轮未收到新消息)"
            )
            # followers & followings
            current_social_network: list[SocialRelation] = await self.memory.status.get(
                "social_network", []
            )
            followers = [
                connection.target_id
                for connection in current_social_network
                if connection.kind == RelationType.FOLLOWER
            ]
            followings = [
                connection.target_id
                for connection in current_social_network
                if connection.kind == RelationType.FOLLOWING
            ]
            # Format intervention history
            intervention_details_list = []
            for hist_item in self.intervention_history:
                intervention_type = hist_item.get("intervention_type", "未知干预")
                details_str_parts = []
                if "post_id" in hist_item.get("details", {}):
                    details_str_parts.append(
                        f"帖子ID:{hist_item['details']['post_id']}"
                    )
                details_combined = ", ".join(details_str_parts)
                intervention_details_list.append(
                    f"第{hist_item.get('round','N/A')}轮:{intervention_type}({details_combined or '无详情'})"
                )
            current_intervention_history_str = (
                "; ".join(intervention_details_list) or "(无干预记录)"
            )

            # Update history summary if needed
            status_mem = self.memory.status
            gender = await status_mem.get("gender", "")
            age_group = await status_mem.get("age_group", "")
            occupation = await status_mem.get("occupation", "")
            preference = await status_mem.get("message_propagation_preference", "")
            if self._current_messages or self.history_summary == "(空)":
                await self.summary_prompt.format(
                    gender=gender,
                    age_group=age_group,
                    occupation=occupation,
                    preference_appendix=self.preference_appendix.get(preference, ""),
                    history_summary=(
                        self.history_summary
                        if self.history_summary != "(空)"
                        else "这是您第一次接触信息。"
                    ),
                    new_messages=new_msgs_str,
                )
                self.history_summary = await self.llm.atext_request(
                    self.summary_prompt.to_dialog(),
                )
                await self.agent.memory.stream.add_social(
                    description=self.history_summary,
                )
            self._current_messages.clear()
            status_mem = self.memory.status
            gender = await status_mem.get("gender", "")
            age_group = await status_mem.get("age_group", "")
            occupation = await status_mem.get("occupation", "")
            preference = await status_mem.get("message_propagation_preference", "")
            # Generate public post
            await self.public_post_prompt.format(
                gender=gender,
                age_group=age_group,
                occupation=occupation,
                preference_appendix=self.preference_appendix.get(preference, ""),
                message_summary=self.history_summary,
                intervention_history=current_intervention_history_str,
            )
            public_post_text = await self.llm.atext_request(
                self.public_post_prompt.to_dialog(),
            )

            # Get public receivers
            public_receivers = []
            if public_post_text:
                public_receivers = random.sample(
                    followers,
                    k=min(len(followers), self.max_visible_followers),
                )
            if public_receivers and public_post_text:
                tasks = [
                    self.agent.send_message_to_agent(
                        to_agent_id=receiver,
                        content=public_post_text,
                    )
                    for receiver in public_receivers
                ]
                await asyncio.gather(*tasks)
            mutuals = list(set(followers) & set(followings))
            # Handle private interactions
            private_chat_targets = random.sample(
                mutuals,
                k=min(len(mutuals), self.max_private_chats),
            )

            # friends info
            friends_info: dict[int, dict[str, Any]] = await self.memory.status.get(
                "friends_info", {}
            )

            if private_chat_targets:
                tasks = []
                status_mem = self.memory.status
                gender = await status_mem.get("gender", "")
                age_group = await status_mem.get("age_group", "")
                occupation = await status_mem.get("occupation", "")
                preference = await status_mem.get("message_propagation_preference", "")
                for target_id in private_chat_targets:
                    private_chat_text = await self.private_chat_prompt.format(
                        gender=gender,
                        age_group=age_group,
                        occupation=occupation,
                        preference_appendix=self.preference_appendix.get(
                            preference, ""
                        ),
                        message_summary=self.history_summary,
                        intervention_history=current_intervention_history_str,
                        friend_gender=friends_info.get(target_id, {}).get("gender", ""),
                        friend_age_group=friends_info.get(target_id, {}).get("age", ""),
                        friend_occupation=friends_info.get(target_id, {}).get(
                            "occupation", ""
                        ),
                    )
                    tasks.append(
                        self.agent.send_message_to_agent(
                            to_agent_id=target_id,
                            content=private_chat_text,
                        )
                    )
                results = await asyncio.gather(*tasks)
            # Calculate token consumption
            consumption_end = (
                self.llm.prompt_tokens_used + self.llm.completion_tokens_used
            )
            self.token_consumption += consumption_end - consumption_start

            return EnvCitizenBlockOutput(
                success=True,
                evaluation="Success",
                consumed_time=time.time() - start_time,
            )

        except Exception as e:
            get_logger().error(f"Social interaction failed: {e}")
            return EnvCitizenBlockOutput(
                success=False,
                evaluation="Failed to complete social interaction with default behavior",
                consumed_time=time.time() - start_time,
            )
