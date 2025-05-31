import asyncio
import random
import time
from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, cast

import jsonc
from agentsociety.agent import (AgentToolbox, Block, StatusAttribute,
                                SupervisorBase)
from agentsociety.agent.prompt import FormatPrompt
from agentsociety.llm import LLM
from agentsociety.memory import Memory
from agentsociety.memory.const import RelationType, SocialRelation
from agentsociety.message import Message, MessageKind
from openai import OpenAIError

from .sensing_api import InterventionType, SensingAPI
from .sharing_params import SupervisorConfig, SupervisorContext

DEFAULT_INTERVENTION_QUOTAS: dict[InterventionType, dict[str, int]] = {
    InterventionType.DELETE_POST: {"per_round": 10, "global": -1},  # -1 表示无全局限制
    InterventionType.PERSUADE_AGENT: {"per_round": 10, "global": -1},
    InterventionType.REMOVE_FOLLOWER: {"per_round": -1, "global": 100},
    InterventionType.BAN_AGENT: {"per_round": -1, "global": 20},
}


def evaluate_logic_expression(expression: str, condition_values: list[bool]) -> bool:
    """
    解析简单的逻辑表达式。
    表达式中的数字（从1开始）对应 condition_values 列表中的索引（需减1）。
    支持 &, |, !, (, ).
    """
    if not expression:
        return True  # 如果没有表达式，默认为True

    # 将数字替换为 condition_values 中的实际布尔值
    for i in range(len(condition_values)):
        expression = expression.replace(str(i + 1), f"c[{i}]")

    # 替换逻辑操作符为python关键字
    expression = expression.replace("&", "and").replace("|", "or").replace("!", "not ")

    try:
        return bool(eval(expression, {"__builtins__": {}}, {"c": condition_values}))
    except Exception as e:
        print(f"[逻辑表达式] 解析 '{expression}' 失败: {e}. 默认返回 False.")
        return False


class RelationNode:
    def __init__(self, id: int):
        self.id = id
        self.following = set()
        self.followers = set()


class RelationNetwork:
    def __init__(self, social_network: list[SocialRelation]):
        self.nodes: dict[int, RelationNode] = {}
        self.degrees: dict[int, int] = {}  # 添加度数统计

        # 遍历社交网络字典
        for connection in social_network:
            agent_id: int = int(connection.source_id)  # type: ignore
            target_id = int(connection.target_id)
            # 初始化节点
            if agent_id not in self.nodes:
                self.nodes[agent_id] = RelationNode(agent_id)
                self.degrees[agent_id] = 0
            if target_id not in self.nodes:
                self.nodes[target_id] = RelationNode(target_id)
                self.degrees[target_id] = 0
            if connection.kind == RelationType.FOLLOWER:
                self.nodes[agent_id].followers.add(target_id)
                self.degrees[agent_id] += 1
                self.degrees[target_id] += 1
            elif connection.kind == RelationType.FOLLOWING:
                self.nodes[agent_id].following.add(target_id)
                self.degrees[agent_id] += 1
                self.degrees[target_id] += 1
            else:
                raise ValueError(f"Invalid connection type: {connection.kind}")

    def following(self, node_id: int) -> Set[int]:
        node_id = int(node_id)
        if node_id not in self.nodes:
            return set()
        return self.nodes[node_id].following

    def followers(self, node_id: int) -> Set[int]:
        node_id = int(node_id)
        if node_id not in self.nodes:
            return set()
        return self.nodes[node_id].followers

    def get_mutual_followers(self, node_id: int) -> list[int]:
        """获取与指定节点互相关注的用户列表"""
        if node_id not in self.nodes:
            return []
        followers = self.nodes[node_id].followers
        following = self.nodes[node_id].following
        return list(followers & following)

    def sample_followers_for_post(self, node_id: int, k: int) -> list[int]:
        """随机采样k个关注者用于发帖"""
        followers = list(self.followers(node_id))
        if not followers:
            return []
        if len(followers) <= k:
            return followers
        return random.sample(followers, k)

    def weighted_sample_nodes_by_degree(
        self, k: int, exclude: Optional[Set[int]] = None
    ) -> list[int]:
        """按度数加权采样k个不同节点"""
        exclude = exclude or set()
        candidates = [
            (node, deg) for node, deg in self.degrees.items() if node not in exclude
        ]
        if not candidates:
            return []
        nodes, weights = zip(*candidates)
        selected = random.choices(nodes, weights=weights, k=k)
        # 保持顺序去重
        selected = list(dict.fromkeys(selected))
        # 如不足k个，再随机补
        while len(selected) < k and len(selected) < len(nodes):
            extra = random.choice(nodes)
            if extra not in selected:
                selected.append(extra)
        return selected

    def get_degree(self, node_id: int) -> int:
        """获取指定节点的度数"""
        return self.degrees.get(node_id, 0)

    def get_network_structure(self) -> dict[str, list[Any]]:
        """获取当前网络的结构信息

        Returns:
            dict: 包含以下键值对:
                - nodes: list[int] - 所有节点的ID列表
                - edges: list[list[int]] - 所有边的列表，每个边为 [source_id, target_id]
        """
        nodes = list(self.degrees.keys())  # 所有有度数的节点
        edges = []
        # 从 following 构建边列表 (source -> target)
        for source_id, node in self.nodes.items():
            for target_id in node.following:
                edges.append([source_id, target_id])
        return {"nodes": nodes, "edges": edges}


class BDSC2025Supervisor(SupervisorBase):
    ParamsType = SupervisorConfig  # type: ignore
    BlockOutputType = Any  # type: ignore
    Context = SupervisorContext  # type: ignore

    def __init__(
        self,
        id: int,
        name: str,
        toolbox: AgentToolbox,
        memory: Memory,
        agent_params: Optional[Any] = None,
        blocks: Optional[list[Block]] = None,
    ):
        super().__init__(id, name, toolbox, memory, agent_params, blocks)
        self.enable_intervention = True
        self.max_process_message_per_round = 50
        self.sensing_api = SensingAPI(intervention_quotas=DEFAULT_INTERVENTION_QUOTAS)
        self.sensing_api._set_supervisor(self)
        self.current_round_number = 0
        self.total_simulation_rounds = 20
        self.rumor_topic_description = []
        self.post_id_counter = 0
        self.max_retry_times = 10
        self.messages_shorten_length = 10
        self.all_responses: list[str] = []  # 所有历史响应
        self.rumor_spreader_id = 5000
        self.params = cast(SupervisorConfig, self.params)
        self.time_record: dict[int, dict[str, float]] = {}

        # 干预配额相关
        self.intervention_quotas = DEFAULT_INTERVENTION_QUOTAS
        self.current_round_quota_usage: dict[str, int] = {
            "delete_post": 0,
            "persuade_agent": 0,
            "remove_follower": 0,
            "ban_agent": 0,
        }
        self.global_quota_usage: dict[str, int] = {
            "delete_post": 0,
            "persuade_agent": 0,
            "remove_follower": 0,
            "ban_agent": 0,
        }
        self.context = cast(SupervisorContext, self.context)

        # 消息历史相关
        self.global_posts_history: list[dict[str, Any]] = []  # 所有历史帖子
        self.current_round_posts_buffer: list[dict[str, Any]] = []  # 当前轮次的预备帖子

        # 网络结构相关
        self.network: Optional[RelationNetwork] = None
        self.agent_map: dict[int, Any] = {}  # 智能体ID到智能体对象的映射

        # 干预相关
        self.banned_agent_ids: Set[int] = set()  # 被封禁的智能体ID集合
        self.globally_removed_edges: Set[Tuple[int, int]] = set()  # 被永久移除的边集合
        self.current_round_interventions: list[dict[str, Any]] = (
            []
        )  # 当前轮次的干预记录
        self.all_historical_interventions_log: list[dict[str, Any]] = (
            []
        )  # 所有历史干预记录
        self.intervention_stats: dict[InterventionType, dict[str, int]] = {
            InterventionType.BAN_AGENT: {
                "per_round_used": 0,
                "global_used": 0,
            },
            InterventionType.PERSUADE_AGENT: {
                "per_round_used": 0,
                "global_used": 0,
            },
            InterventionType.DELETE_POST: {
                "per_round_used": 0,
                "global_used": 0,
            },
            InterventionType.REMOVE_FOLLOWER: {
                "per_round_used": 0,
                "global_used": 0,
            },
        }

        # 当前轮次的干预结果
        self._current_validation_dict: dict[Message, bool] = {}
        self._current_blocked_agent_ids: list[int] = []
        self._current_blocked_social_edges: list[tuple[int, int]] = []
        self._current_persuasion_messages: list[Message] = []

    def get_current_round_number(self) -> int:
        return self.current_round_number

    def get_total_simulation_rounds(self) -> int:
        return self.total_simulation_rounds

    def _check_and_update_quota(self, intervention_type: str) -> bool:
        """检查并更新干预配额

        Args:
            intervention_type: 干预类型，必须是 intervention_quotas 中的键

        Returns:
            bool: 是否可以使用配额
        """
        if intervention_type not in self.intervention_quotas:
            return False

        quota = self.intervention_quotas[intervention_type]
        current_usage = self.current_round_quota_usage[intervention_type]
        global_usage = self.global_quota_usage[intervention_type]

        # 检查每轮配额
        if quota["per_round"] != -1 and current_usage >= quota["per_round"]:
            return False

        # 检查全局配额
        if quota["global"] != -1 and global_usage >= quota["global"]:
            return False

        # 更新使用量
        self.current_round_quota_usage[intervention_type] += 1
        self.global_quota_usage[intervention_type] += 1
        return True

    async def forward(
        self,
        current_round_messages: list[Message],
    ) -> tuple[
        dict[Message, bool],
        list[Message],
    ]:
        """
        处理当前轮次的消息，进行验证和干预

        Args:
            current_round_messages: 当前轮次的消息列表，每个元素为 (sender_id, receiver_id, content) 的元组
            llm: LLM实例，用于内容验证

        Returns:
            validation_dict: 消息验证结果字典，key为消息元组，value为是否通过验证
            persuasion_messages: 劝导消息列表
        """
        self.time_record[self.current_round_number] = {
            "start_time": time.time(),
        }
        # 初始化网络结构
        if self.network is None:
            # agent id -> following & followers
            social_network_from_memory: list[SocialRelation] = (
                await self.memory.status.get("social_network", [])
            )
            # if len(social_network_from_memory) == 0:
            #     raise ValueError("Social network is empty")
            self.network = RelationNetwork(social_network_from_memory)
        assert self.network is not None, "Network is not initialized"
        # 清空当前轮次的缓冲区和干预记录
        added_sender_and_msg: set[tuple[int, str]] = set()
        identifier_to_post: dict[tuple[int, str], dict[str, Any]] = {}
        for msg in current_round_messages:
            content = msg.payload["content"]
            sender_id = msg.from_id
            receiver_id = msg.to_id
            if sender_id is None or receiver_id is None:
                continue
            identifier = (sender_id, content)
            if identifier in added_sender_and_msg:
                post = identifier_to_post[identifier]
                post["original_intended_receiver_ids"].append(receiver_id)
                continue
            else:
                added_sender_and_msg.add(identifier)
                identifier_to_post[identifier] = {
                    "sender_id": sender_id,
                    "post_id": f"post_{self.post_id_counter}",
                    "receiver_id": receiver_id,
                    "message": msg,
                    "content": content,
                    "round": self.current_round_number,
                    "original_intended_receiver_ids": [receiver_id],
                }
            self.post_id_counter += 1
        self.current_round_posts_buffer = [
            v
            for v in identifier_to_post.values()
            if len(v["original_intended_receiver_ids"]) > 1
        ]
        self.current_round_interventions = []

        # 重置当前轮次的配额使用量
        for intervention_type in self.current_round_quota_usage:
            self.current_round_quota_usage[intervention_type] = 0

        # 初始化当前轮次的干预结果
        self._current_validation_dict: dict[Message, bool] = {}
        self._current_blocked_agent_ids: list[int] = []
        self._current_blocked_social_edges: list[tuple[int, int]] = []
        self._current_persuasion_messages: list[Message] = []

        # 更新context
        # round number
        self.context.current_round_number = self.current_round_number
        # posts
        self.context.current_round_posts = [
            {
                "sender_id": post["sender_id"],
                "post_id": post["post_id"],
                "content": post["content"],
                "original_intended_receiver_ids": post[
                    "original_intended_receiver_ids"
                ],
            }
            for post in self.current_round_posts_buffer
        ]
        # posts
        self.context.current_round_posts = []
        # network structure
        self.context.current_round_post_followers = {  # type: ignore
            post["sender_id"]: self.network.followers(post["sender_id"])
            for post in self.current_round_posts_buffer
            if post["sender_id"] != self.rumor_spreader_id
        }
        self.context.current_round_post_following = {  # type: ignore
            post["sender_id"]: self.network.following(post["sender_id"])
            for post in self.current_round_posts_buffer
            if post["sender_id"] != self.rumor_spreader_id
        }
        # ban agent
        self.context.current_round_ban_agent_usage = 0
        self.context.global_ban_agent_usage = self.global_quota_usage["ban_agent"]
        self.context.current_round_ban_agent_quota = self.intervention_quotas[
            InterventionType.BAN_AGENT
        ]["per_round"]
        self.context.global_ban_agent_quota = self.intervention_quotas[
            InterventionType.BAN_AGENT
        ]["global"]
        # persuade agent
        self.context.current_round_persuade_agent_usage = 0
        self.context.global_persuade_agent_usage = self.global_quota_usage[
            InterventionType.PERSUADE_AGENT
        ]
        self.context.current_round_persuade_agent_quota = self.intervention_quotas[
            InterventionType.PERSUADE_AGENT
        ]["per_round"]
        self.context.global_persuade_agent_quota = self.intervention_quotas[
            InterventionType.PERSUADE_AGENT
        ]["global"]
        # delete post
        self.context.current_round_delete_post_usage = 0
        self.context.global_delete_post_usage = self.global_quota_usage[
            InterventionType.DELETE_POST
        ]
        self.context.current_round_delete_post_quota = self.intervention_quotas[
            InterventionType.DELETE_POST
        ]["per_round"]
        self.context.global_delete_post_quota = self.intervention_quotas[
            InterventionType.DELETE_POST
        ]["global"]
        # remove follower
        self.context.current_round_remove_follower_usage = 0
        self.context.global_remove_follower_usage = self.global_quota_usage[
            InterventionType.REMOVE_FOLLOWER
        ]
        self.context.current_round_remove_follower_quota = self.intervention_quotas[
            InterventionType.REMOVE_FOLLOWER
        ]["per_round"]
        self.context.global_remove_follower_quota = self.intervention_quotas[
            InterventionType.REMOVE_FOLLOWER
        ]["global"]
        # 调用干预方法
        if self.enable_intervention:
            await self.interventions()

        # ATTENTION: dump干预log
        with open(
            f"all_historical_interventions_log.json",
            "w",
        ) as f:
            jsonc.dump(self.all_historical_interventions_log, f)

        with open(
            f"all_historical_responses.json",
            "w",
        ) as f:
            jsonc.dump(self.all_responses, f)

        # 更新时间记录
        self.time_record[self.current_round_number]["end_time"] = time.time()
        # 更新轮次计数
        self.current_round_number += 1
        self.global_posts_history.extend(self.current_round_posts_buffer)
        with open(
            f"time_record.json",
            "w",
        ) as f:
            jsonc.dump(self.time_record, f)
        # 根据ban_agent_ids和_blocked_social_edges，更新_current_validation_dict
        for msg in list(self._current_validation_dict.keys()):
            if msg.from_id in self._current_blocked_agent_ids:
                self._current_validation_dict[msg] = False
            if (msg.from_id, msg.to_id) in self._current_blocked_social_edges:
                self._current_validation_dict[msg] = False
        return (
            self._current_validation_dict,
            self._current_persuasion_messages,
        )

    async def interventions(
        self,
    ) -> None:
        """
        检测并执行干预措施
        """
        assert self.network is not None, "Network is not initialized"
        # 1. 检测阶段 - 检查当前轮次的所有帖子
        all_scores: list[float] = []

        async def _llm_detect_post(post: dict[str, Any]) -> tuple[bool, str]:
            """使用LLM检测单个帖子是否违规。返回 (是否违规, 理由/原始输出)。"""
            prompt = FormatPrompt(
                template=self.params.llm_detection_prompt,
            )
            temporary_context = SupervisorContext(
                # Round number
                current_round_number=self.current_round_number,
                # Current processing message
                current_processing_message=post["content"],
                current_processing_message_sender_id=post["sender_id"],
                current_processing_message_receiver_ids=post[
                    "original_intended_receiver_ids"
                ],
                # Current round posts
                current_round_posts=[
                    {
                        "sender_id": p["sender_id"],
                        "post_id": p["post_id"],
                        "content": p["content"],
                        "original_intended_receiver_ids": p[
                            "original_intended_receiver_ids"
                        ],
                        "is_violating": p.get("is_violating", False),
                    }
                    for p in self.current_round_posts_buffer
                ],
                # Network structure
                current_round_post_followers={
                    p["sender_id"]: (
                        list(self.network.followers(p["sender_id"]))
                        if self.network
                        else []
                    )
                    for p in self.current_round_posts_buffer
                    if p["sender_id"] != self.rumor_spreader_id
                },
                current_round_post_following={
                    p["sender_id"]: (
                        list(self.network.following(p["sender_id"]))
                        if self.network
                        else []
                    )
                    for p in self.current_round_posts_buffer
                    if p["sender_id"] != self.rumor_spreader_id
                },
                # Ban agent quotas and usage
                current_round_ban_agent_usage=self.current_round_quota_usage[
                    "ban_agent"
                ],
                current_round_ban_agent_quota=self.intervention_quotas[
                    InterventionType.BAN_AGENT
                ]["per_round"],
                global_ban_agent_usage=self.global_quota_usage["ban_agent"],
                global_ban_agent_quota=self.intervention_quotas[
                    InterventionType.BAN_AGENT
                ]["global"],
                # Persuade agent quotas and usage
                current_round_persuade_agent_usage=self.current_round_quota_usage[
                    "persuade_agent"
                ],
                current_round_persuade_agent_quota=self.intervention_quotas[
                    InterventionType.PERSUADE_AGENT
                ]["per_round"],
                global_persuade_agent_usage=self.global_quota_usage["persuade_agent"],
                global_persuade_agent_quota=self.intervention_quotas[
                    InterventionType.PERSUADE_AGENT
                ]["global"],
                # Delete post quotas and usage
                current_round_delete_post_usage=self.current_round_quota_usage[
                    "delete_post"
                ],
                current_round_delete_post_quota=self.intervention_quotas[
                    InterventionType.DELETE_POST
                ]["per_round"],
                global_delete_post_usage=self.global_quota_usage["delete_post"],
                global_delete_post_quota=self.intervention_quotas[
                    InterventionType.DELETE_POST
                ]["global"],
                # Remove follower quotas and usage
                current_round_remove_follower_usage=self.current_round_quota_usage[
                    "remove_follower"
                ],
                current_round_remove_follower_quota=self.intervention_quotas[
                    InterventionType.REMOVE_FOLLOWER
                ]["per_round"],
                global_remove_follower_usage=self.global_quota_usage["remove_follower"],
                global_remove_follower_quota=self.intervention_quotas[
                    InterventionType.REMOVE_FOLLOWER
                ]["global"],
                # Current agent info for risk assessment
                current_agent_degree=(
                    self.network.get_degree(post["sender_id"]) if self.network else 0
                ),
                current_agent_offense_summary=str(
                    self.sensing_api.get_agent_historical_offense_summary(
                        post["sender_id"]
                    )
                ),
                current_processing_agent_id=post["sender_id"],
                current_agent_intervention_count=len(
                    [
                        i
                        for i in self.all_historical_interventions_log
                        if i.get("target_agent_id") == post["sender_id"]
                    ]
                ),
            )
            await prompt.format(context=temporary_context)  # type: ignore
            try:
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                response = response.strip().lower()
                if response.startswith("是") or response.startswith("yes"):
                    reason = (
                        response.split("，", 1)[1] if "，" in response else response
                    )
                    return True, reason
                elif response.startswith("否") or response.startswith("no"):
                    return False, response
                else:
                    print(f"[LLM检测] LLM响应格式不明确: '{response}'.默认帖子合规")
                    return False, response
            except Exception as e:
                print(f"[LLM检测] 检测失败: {e}。默认帖子合规")
                return False, str(e)

        def _keyword_detect_post(post_content: str) -> tuple[bool, list[str]]:
            """使用关键词检测单个帖子。返回 (是否违规, 命中的关键词列表)。"""
            content_lower = post_content.lower()

            # 检查排除词
            for ex_word in self.params.keyword_detection_exclude_words:
                if ex_word.lower() in content_lower:
                    return False, []  # 包含排除词，直接判定为不违规

            # 检查关键词
            matched_keywords = []
            for kw in self.params.keyword_detection_keywords:
                if kw.lower() in content_lower:
                    matched_keywords.append(kw)

            # 根据配置决定是否将关键词存在视为违规
            if (
                matched_keywords
                and self.params.keyword_detection_violation_if_keywords_present
            ):
                return True, matched_keywords
            elif (
                not matched_keywords
                and not self.params.keyword_detection_violation_if_keywords_present
            ):
                return True, []
            return False, matched_keywords

        async def is_violating(post: dict[str, Any]) -> bool:
            # 1. LLM检测
            llm_violating, llm_reason = await _llm_detect_post(post)

            # 2. 关键词检测
            kw_violating, kw_matched = _keyword_detect_post(post["content"])

            # 3. 根据逻辑表达式组合检测结果
            condition_values = [llm_violating, kw_violating]
            final_is_violating = evaluate_logic_expression(
                self.params.detection_logic_expression, condition_values
            )

            # 如果检测结果为违规，直接返回高分
            return final_is_violating

        print(f"Processing {len(self.current_round_posts_buffer)} posts")
        tasks = [is_violating(post) for post in self.current_round_posts_buffer]
        all_violating = await asyncio.gather(*tasks)
        for is_violate, post in zip(all_violating, self.current_round_posts_buffer):
            post["is_violating"] = is_violate
        # 2. 干预阶段
        self.context.current_round_posts = [
            {
                "sender_id": post["sender_id"],
                "post_id": post["post_id"],
                "content": post["content"],
                "original_intended_receiver_ids": post[
                    "original_intended_receiver_ids"
                ],
                "is_violating": post["is_violating"],
            }
            for post in self.current_round_posts_buffer
        ]
        # 更新当前轮次违规情况

        # 将四个监管措施封装成独立的异步函数
        async def delete_post_intervention_async():
            if not self.params.delete_post_enabled:
                return
            # meet quota
            if not self._check_and_update_quota("delete_post"):
                return
            assert self.network is not None, "Network is not initialized"
            for _ in range(self.max_retry_times):
                try:
                    # 应用删帖策略
                    posts_to_delete = [
                        p
                        for p in self.current_round_posts_buffer
                        if p["is_violating"]
                        and p["sender_id"] != self.rumor_spreader_id
                    ]
                    if len(posts_to_delete) == 0:
                        return

                    # 并行执行LLM检测和关键词检测
                    async def evaluate_post(post):
                        llm_violating, _ = await _llm_detect_post(post)
                        kw_violating, _ = _keyword_detect_post(post["content"])
                        return post, llm_violating or kw_violating

                    evaluation_tasks = [evaluate_post(post) for post in posts_to_delete]
                    evaluation_results = await asyncio.gather(*evaluation_tasks)

                    # 过滤出违规的帖子
                    posts_to_delete = [
                        post
                        for post, is_violating in evaluation_results
                        if is_violating
                    ]

                    # 按策略排序
                    strategy = self.params.delete_post_priority_strategy
                    if strategy == "sender_degree_high":
                        posts_to_delete.sort(
                            key=lambda p: (
                                self.network.get_degree(p["sender_id"])
                                if self.network
                                else 0
                            ),
                            reverse=True,
                        )
                    elif strategy == "receiver_degree_high":

                        def get_avg_receiver_degree(post):
                            receivers = post.get("original_intended_receiver_ids", [])
                            if not receivers:
                                return 0
                            return (
                                sum(self.network.get_degree(r_id) for r_id in receivers)
                                / len(receivers)
                                if self.network
                                else 0
                            )

                        posts_to_delete.sort(key=get_avg_receiver_degree, reverse=True)
                    elif strategy == "sender_least_intervened":
                        posts_to_delete.sort(
                            key=lambda p: len(
                                [
                                    i
                                    for i in self.all_historical_interventions_log
                                    if i.get("target_agent_id") == p["sender_id"]
                                ]
                            )
                        )
                    elif strategy == "random":
                        random.shuffle(posts_to_delete)

                    # 执行删帖
                    for post in posts_to_delete:
                        self.delete_post_intervention(post["post_id"], "低分帖子")
                    break
                except OpenAIError as e:
                    self.context.current_round_posts = self.context.current_round_posts[
                        : -self.messages_shorten_length
                    ]
                except Exception as e:
                    print(f"Delete post error: {e}")

        async def persuade_agent_intervention_async():
            if not self.params.persuade_agent_enabled:
                return
            # meet quota
            if not self._check_and_update_quota("persuade_agent"):
                return
            network = cast(RelationNetwork, self.network)
            for _ in range(self.max_retry_times):
                try:
                    agents_to_persuade = list(network.nodes.keys())

                    # 并行评估劝说条件
                    async def evaluate_agent_conditions(agent_id: int) -> Optional[int]:
                        cond_values: list[bool] = []
                        # 条件1: agent_sent_rumor_this_round
                        if (
                            self.params.persuade_agent_condition_sent_rumor
                            in self.params.persuade_agent_trigger_conditions_expression
                        ):
                            cond_values.append(
                                bool(
                                    [
                                        p
                                        for p in self.current_round_posts_buffer
                                        if p["sender_id"] == agent_id
                                    ]
                                )
                            )
                        else:
                            cond_values.append(False)

                        # 条件2: agent_degree_top_k
                        if (
                            "2"
                            in self.params.persuade_agent_trigger_conditions_expression
                        ):
                            assert (
                                self.network is not None
                            ), "Network is not initialized"
                            k = self.params.persuade_agent_condition_degree_top_k
                            top_k_nodes = sorted(
                                [
                                    (aid, self.network.get_degree(aid))
                                    for aid in self.network.nodes.keys()
                                ],
                                key=lambda x: x[1],
                                reverse=True,
                            )[:k]
                            cond_values.append(
                                agent_id in [aid for aid, _ in top_k_nodes]
                            )
                        else:
                            cond_values.append(False)

                        # 条件3: agent_never_persuaded_before
                        if (
                            self.params.persuade_agent_condition_never_persuaded
                            in self.params.persuade_agent_trigger_conditions_expression
                        ):
                            history = [
                                i
                                for i in self.all_historical_interventions_log
                                if i.get("target_agent_id") == agent_id
                            ]
                            was_persuaded = any(
                                item.get("type") == "persuade_agent" for item in history
                            )
                            cond_values.append(not was_persuaded)
                        else:
                            cond_values.append(False)

                        if evaluate_logic_expression(
                            self.params.persuade_agent_trigger_conditions_expression,
                            cond_values,
                        ):
                            return agent_id
                        return None

                    evaluation_tasks = [
                        evaluate_agent_conditions(agent_id)
                        for agent_id in agents_to_persuade
                    ]
                    evaluation_results: list[Optional[int]] = await asyncio.gather(
                        *evaluation_tasks
                    )
                    agents_to_persuade_filtered = [
                        result for result in evaluation_results if result is not None
                    ]

                    # 按策略排序
                    strategy = self.params.persuade_agent_priority_strategy
                    if strategy == "degree_high":
                        agents_to_persuade_filtered.sort(
                            key=lambda x: (network.get_degree(x)),
                            reverse=True,
                        )
                    elif strategy == "most_violated_this_round":
                        agents_to_persuade_filtered.sort(
                            key=lambda x: len(
                                [
                                    p
                                    for p in self.current_round_posts_buffer
                                    if p["sender_id"] == x
                                ]
                            ),
                            reverse=True,
                        )
                    elif strategy == "least_intervened":
                        agents_to_persuade_filtered.sort(
                            key=lambda x: len(
                                [
                                    i
                                    for i in self.all_historical_interventions_log
                                    if i.get("target_agent_id") == x
                                ]
                            )
                        )
                    elif strategy == "random":
                        random.shuffle(agents_to_persuade_filtered)

                    # 执行劝说
                    for agent_id in agents_to_persuade_filtered:
                        self.persuade_agent_intervention(
                            agent_id, self.params.persuade_agent_content
                        )
                    break
                except OpenAIError as e:
                    self.context.current_round_posts = self.context.current_round_posts[
                        : -self.messages_shorten_length
                    ]
                except Exception as e:
                    print(f"Persuade agent error: {e}")

        async def remove_follower_intervention_async():
            if not self.params.remove_follower_enabled:
                return
            # meet quota
            if not self._check_and_update_quota("remove_follower"):
                return
            assert self.network is not None, "Network is not initialized"
            for _ in range(self.max_retry_times):
                try:
                    # 应用移除关注者策略
                    edges_to_remove = [
                        (int(edge[0]), int(edge[1]))
                        for edge in self.globally_removed_edges
                    ]

                    # 并行评估移除关注者条件
                    async def evaluate_edge_conditions(edge):
                        follower_id, following_id = edge
                        cond_values = []
                        # 条件1: both_agents_high_risk_llm
                        if (
                            "1"
                            in self.params.remove_follower_trigger_conditions_expression
                        ):
                            # 并行评估两个智能体的风险
                            summary_s = (
                                self.sensing_api.get_agent_historical_offense_summary(
                                    following_id
                                )
                            )
                            summary_t = (
                                self.sensing_api.get_agent_historical_offense_summary(
                                    follower_id
                                )
                            )

                            following_is_high_risk, follower_is_high_risk = (
                                await asyncio.gather(
                                    self._call_llm_for_risk_assessment(
                                        following_id,
                                        self.params.remove_follower_condition_high_risk_prompt,
                                        str(summary_s),
                                    ),
                                    self._call_llm_for_risk_assessment(
                                        follower_id,
                                        self.params.remove_follower_condition_high_risk_prompt,
                                        str(summary_t),
                                    ),
                                )
                            )
                            cond_values.append(
                                following_is_high_risk and follower_is_high_risk
                            )
                        else:
                            cond_values.append(False)

                        # 条件2: both_agents_degree_over_threshold
                        if (
                            "2"
                            in self.params.remove_follower_trigger_conditions_expression
                        ):
                            threshold = (
                                self.params.remove_follower_condition_degree_threshold
                            )
                            deg_s = (
                                self.network.get_degree(following_id)
                                if self.network
                                else 0
                            )
                            deg_t = (
                                self.network.get_degree(follower_id)
                                if self.network
                                else 0
                            )
                            cond_values.append(
                                (deg_s > threshold) and (deg_t > threshold)
                            )
                        else:
                            cond_values.append(False)

                        # 条件3: edge_rumor_traffic_over_threshold
                        if (
                            "3"
                            in self.params.remove_follower_trigger_conditions_expression
                        ):
                            threshold = (
                                self.params.remove_follower_condition_traffic_threshold
                            )
                            traffic = len(
                                [
                                    p
                                    for p in self.global_posts_history
                                    if p["sender_id"] == following_id
                                    and follower_id
                                    in p["original_intended_receiver_ids"]
                                ]
                            )
                            cond_values.append(traffic > threshold)
                        else:
                            cond_values.append(False)

                        if evaluate_logic_expression(
                            self.params.remove_follower_trigger_conditions_expression,
                            cond_values,
                        ):
                            return edge
                        return None

                    evaluation_tasks = [
                        evaluate_edge_conditions(edge) for edge in edges_to_remove
                    ]
                    evaluation_results = await asyncio.gather(*evaluation_tasks)
                    edges_to_remove_filtered = [
                        result for result in evaluation_results if result is not None
                    ]

                    # 按风险分数排序
                    edges_to_remove_filtered.sort(
                        key=lambda edge: self._evaluate_edge_risk(edge), reverse=True
                    )

                    # 执行移除关注者
                    for follower_id, following_id in edges_to_remove_filtered:
                        self.remove_follower_intervention(follower_id, following_id)
                    break
                except OpenAIError as e:
                    self.context.current_round_posts = self.context.current_round_posts[
                        : -self.messages_shorten_length
                    ]
                except Exception as e:
                    print(f"Remove follower error: {e}")

        async def ban_agent_intervention_async():
            if not self.params.ban_agent_enabled:
                return
            # meet quota
            if not self._check_and_update_quota("ban_agent"):
                return
            assert self.network is not None, "Network is not initialized"
            for _ in range(self.max_retry_times):
                try:
                    # 所有的agent
                    agents_to_ban = list(self.network.nodes.keys())

                    # 并行评估封禁条件
                    async def evaluate_ban_conditions(agent_id):
                        if agent_id in self.banned_agent_ids:
                            return None

                        cond_values = []
                        summary = self.sensing_api.get_agent_historical_offense_summary(
                            agent_id
                        )
                        inter_history = [
                            i
                            for i in self.all_historical_interventions_log
                            if i.get("target_agent_id") == agent_id
                        ]

                        # 条件1: agent_total_violations_over_threshold
                        if "1" in self.params.ban_agent_trigger_conditions_expression:
                            threshold = (
                                self.params.ban_agent_condition_violations_threshold
                            )
                            total_violations = len(
                                [
                                    p
                                    for p in self.current_round_posts_buffer
                                    if p["sender_id"] == agent_id and p["is_violating"]
                                ]
                            )
                            cond_values.append(total_violations > threshold)
                        else:
                            cond_values.append(False)

                        # 条件2: agent_violated_after_many_interventions
                        if "2" in self.params.ban_agent_trigger_conditions_expression:
                            inter_threshold = (
                                self.params.ban_agent_condition_intervention_threshold
                            )
                            sent_rumor_this_round = bool(
                                [
                                    p
                                    for p in self.current_round_posts_buffer
                                    if p["sender_id"] == agent_id
                                ]
                            )
                            total_interventions = len(inter_history)
                            cond_values.append(
                                sent_rumor_this_round
                                and (total_interventions > inter_threshold)
                            )
                        else:
                            cond_values.append(False)

                        # 条件3: agent_high_risk_llm
                        if "3" in self.params.ban_agent_trigger_conditions_expression:
                            cond_values.append(
                                await self._call_llm_for_risk_assessment(
                                    agent_id,
                                    self.params.ban_agent_condition_high_risk_prompt,
                                    str(summary),
                                    len(inter_history),
                                )
                            )
                        else:
                            cond_values.append(False)

                        if evaluate_logic_expression(
                            self.params.ban_agent_trigger_conditions_expression,
                            cond_values,
                        ):
                            return agent_id
                        return None

                    evaluation_tasks = [
                        evaluate_ban_conditions(agent_id) for agent_id in agents_to_ban
                    ]
                    evaluation_results = await asyncio.gather(*evaluation_tasks)
                    agents_to_ban_filtered = [
                        result for result in evaluation_results if result is not None
                    ]

                    # 按风险分数排序
                    agents_to_ban_filtered.sort(
                        key=lambda agent_id: self._evaluate_agent_risk(agent_id),
                        reverse=True,
                    )

                    # 执行封禁
                    for agent_id in agents_to_ban_filtered:
                        self.ban_agent_intervention(agent_id, "低分帖子")
                    break
                except OpenAIError as e:
                    self.context.current_round_posts = self.context.current_round_posts[
                        : -self.messages_shorten_length
                    ]
                except Exception as e:
                    print(f"Ban agent error: {e}")

        # 并行执行四个监管措施
        await asyncio.gather(
            delete_post_intervention_async(),
            persuade_agent_intervention_async(),
            remove_follower_intervention_async(),
            ban_agent_intervention_async(),
        )

    def _evaluate_edge_risk(self, edge: tuple[int, int]) -> float:
        """评估边的风险分数"""
        follower_id, following_id = edge
        # 获取两个节点的度数
        follower_degree = self.network.get_degree(follower_id) if self.network else 0
        following_degree = self.network.get_degree(following_id) if self.network else 0
        # 获取历史违规记录
        follower_violations = len(
            [
                p
                for p in self.global_posts_history
                if p["sender_id"] == follower_id and p["is_violating"]
            ]
        )
        following_violations = len(
            [
                p
                for p in self.global_posts_history
                if p["sender_id"] == following_id and p["is_violating"]
            ]
        )
        # 计算综合风险分数
        return (follower_degree + following_degree) * (
            follower_violations + following_violations
        )

    def _evaluate_agent_risk(self, agent_id: int) -> float:
        """评估智能体的风险分数"""
        # 获取智能体度数
        degree = self.network.get_degree(agent_id) if self.network else 0
        # 获取历史违规记录
        violations = len(
            [
                p
                for p in self.global_posts_history
                if p["sender_id"] == agent_id and p["is_violating"]
            ]
        )
        # 获取历史干预记录
        interventions = len(
            [
                i
                for i in self.all_historical_interventions_log
                if i.get("target_agent_id") == agent_id
            ]
        )
        # 计算综合风险分数
        return degree * (violations + interventions)

    def ban_agent_intervention(self, agent_id: int, reason: str) -> bool:
        """
        封禁指定智能体

        Args:
            agent_id: 要封禁的智能体ID
            reason: 封禁原因

        Returns:
            bool: 是否成功封禁
        """
        if agent_id == self.rumor_spreader_id:
            return False
        if agent_id in self.banned_agent_ids:
            return False

        # 检查配额
        if not self._check_and_update_quota("ban_agent"):
            return False

        self.banned_agent_ids.add(agent_id)
        self._current_blocked_agent_ids.append(agent_id)
        day, t = self.environment.get_datetime()
        self._current_persuasion_messages.append(
            Message(
                from_id=self.id,
                to_id=agent_id,
                day=day,
                t=t,
                kind=MessageKind.AGENT_CHAT,
                payload={"type": "agent_banned", "content": reason},
            )
        )

        # 更新所有涉及该智能体的消息的验证结果
        for msg, is_valid in list(self._current_validation_dict.items()):
            if msg.from_id == agent_id or msg.to_id == agent_id:
                self._current_validation_dict[msg] = False

        # 记录干预
        intervention = {
            "round": self.current_round_number,
            "type": InterventionType.BAN_AGENT,
            "target_agent_id": agent_id,
            "reason": reason,
        }
        self.current_round_interventions.append(intervention)
        self.all_historical_interventions_log.append(intervention)

        # 更新干预统计
        if "agent_banned" not in self.intervention_stats:
            self.intervention_stats[InterventionType.BAN_AGENT] = {
                "per_round_used": 0,
                "global_used": 0,
            }
        self.intervention_stats[InterventionType.BAN_AGENT]["per_round_used"] += 1
        self.intervention_stats[InterventionType.BAN_AGENT]["global_used"] += 1
        self.context.current_round_ban_agent_usage += 1
        return True

    def persuade_agent_intervention(self, target_agent_id: int, message: str) -> bool:
        """
        向指定智能体发送劝导消息

        Args:
            target_agent_id: 目标智能体ID
            message: 劝导消息内容

        Returns:
            bool: 是否成功发送
        """
        # 检查配额
        if not self._check_and_update_quota("persuade_agent"):
            return False
        
        
        day, t = self.environment.get_datetime()

        self._current_persuasion_messages.append(
            Message(
                from_id=self.id,
                to_id=target_agent_id,
                day=day,
                t=t,
                kind=MessageKind.AGENT_CHAT,
                payload={
                    "type": "persuasion",
                    "content": message,
                    "round": self.current_round_number,
                },
            )
        )

        # 记录干预
        intervention = {
            "round": self.current_round_number,
            "type": InterventionType.PERSUADE_AGENT,
            "target_agent_id": target_agent_id,
            "message": message,
        }
        self.current_round_interventions.append(intervention)
        self.all_historical_interventions_log.append(intervention)

        # 更新干预统计
        if InterventionType.PERSUADE_AGENT not in self.intervention_stats:
            self.intervention_stats[InterventionType.PERSUADE_AGENT] = {
                "per_round_used": 0,
                "global_used": 0,
            }
        self.intervention_stats[InterventionType.PERSUADE_AGENT]["per_round_used"] += 1
        self.intervention_stats[InterventionType.PERSUADE_AGENT]["global_used"] += 1
        self.context.current_round_persuade_agent_usage += 1
        self.context.global_persuade_agent_usage += 1

        return True

    def delete_post_intervention(self, post_id: str, reason: str) -> bool:
        """
        阻止指定的消息

        Args:
            post_id: 消息ID
            reason: 阻止原因

        Returns:
            bool: 是否成功阻止
        """
        # 检查配额
        if not self._check_and_update_quota("delete_post"):
            return False

        # 查找对应的消息
        post = next(
            (p for p in self.current_round_posts_buffer if p["post_id"] == post_id),
            None,
        )
        if post is None:
            return False
        
        day, t = self.environment.get_datetime()
        self._current_persuasion_messages.append(
            Message(
                from_id=self.id,
                to_id=post["sender_id"],
                day=day,
                t=t,
                kind=MessageKind.AGENT_CHAT,
                payload={"type": "post_deleted", "post_id": post_id},
            )
        )

        # 更新消息的验证结果
        for msg, is_valid in list(self._current_validation_dict.items()):
            if (
                msg.from_id == post["sender_id"]
                and msg.to_id in post["original_intended_receiver_ids"]
                and msg.payload["content"] == post["content"]
            ):
                self._current_validation_dict[msg] = False

        # 记录干预
        intervention = {
            "round": self.current_round_number,
            "type": InterventionType.DELETE_POST,
            "post_id": post_id,
            "reason": reason,
        }
        self.current_round_interventions.append(intervention)
        self.all_historical_interventions_log.append(intervention)

        # 更新干预统计
        if InterventionType.DELETE_POST not in self.intervention_stats:
            self.intervention_stats[InterventionType.DELETE_POST] = {
                "per_round_used": 0,
                "global_used": 0,
            }
        self.intervention_stats[InterventionType.DELETE_POST]["per_round_used"] += 1
        self.intervention_stats[InterventionType.DELETE_POST]["global_used"] += 1
        self.context.current_round_delete_post_usage += 1
        self.context.global_delete_post_usage += 1

        return True

    def remove_follower_intervention(
        self, follower_to_remove_id: int, following_agent_id: int
    ) -> bool:
        """
        移除指定智能体的关注者

        Args:
            follower_to_remove_id: 要移除的关注者ID
            following_agent_id: 被关注的智能体ID

        Returns:
            bool: 是否成功移除关注者
        """
        edge = (following_agent_id, follower_to_remove_id)
        if edge in self.globally_removed_edges:
            return False
        if following_agent_id == self.rumor_spreader_id:
            return False
        if follower_to_remove_id == self.rumor_spreader_id:
            return False
        # 检查配额
        if not self._check_and_update_quota("remove_follower"):
            return False

        self.globally_removed_edges.add(edge)
        self._current_blocked_social_edges.append(edge)

        # 更新涉及该边的消息的验证结果
        for msg, is_valid in list(self._current_validation_dict.items()):
            if msg.from_id == following_agent_id and msg.to_id == follower_to_remove_id:
                self._current_validation_dict[msg] = False

        # 添加message给智能体 移除对应的follower和following
        day, t = self.environment.get_datetime()
        self._current_persuasion_messages.append(
            Message(
                from_id=self.id,
                to_id=following_agent_id,
                day=day,
                t=t,
                kind=MessageKind.AGENT_CHAT,
                payload={
                    "type": "remove-follower",
                    "to_remove_id": follower_to_remove_id,
                },
            )
        )
        self._current_persuasion_messages.append(
            Message(
                from_id=self.id,
                to_id=follower_to_remove_id,
                day=day,
                t=t,
                kind=MessageKind.AGENT_CHAT,
                payload={
                    "type": "remove-following",
                    "to_remove_id": following_agent_id,
                },
            )
        )

        # 记录干预
        intervention = {
            "round": self.current_round_number,
            "type": InterventionType.REMOVE_FOLLOWER,
            "following_agent_id": following_agent_id,
            "follower_id": follower_to_remove_id,
        }
        self.current_round_interventions.append(intervention)
        self.all_historical_interventions_log.append(intervention)

        # 更新干预统计
        if InterventionType.REMOVE_FOLLOWER not in self.intervention_stats:
            self.intervention_stats[InterventionType.REMOVE_FOLLOWER] = {
                "per_round_used": 0,
                "global_used": 0,
            }
        self.intervention_stats[InterventionType.REMOVE_FOLLOWER]["per_round_used"] += 1
        self.intervention_stats[InterventionType.REMOVE_FOLLOWER]["global_used"] += 1
        self.context.current_round_remove_follower_usage += 1
        self.context.global_remove_follower_usage += 1
        return True

    async def _call_llm_for_risk_assessment(
        self,
        agent_id: int,
        prompt_template: str,
        offense_summary: str,
        intervention_count: Optional[int] = None,
    ) -> bool:
        """调用LLM进行风险评估，返回True（高风险/应干预）或False。"""
        prompt = FormatPrompt(
            template=prompt_template,
        )
        assert self.network is not None, "Network is not initialized"
        temporary_context = SupervisorContext(
            # Round number
            current_round_number=self.current_round_number,
            # Current round posts
            current_round_posts=[
                {
                    "sender_id": p["sender_id"],
                    "post_id": p["post_id"],
                    "content": p["content"],
                    "original_intended_receiver_ids": p[
                        "original_intended_receiver_ids"
                    ],
                    "is_violating": p.get("is_violating", False),
                }
                for p in self.current_round_posts_buffer
            ],
            # Network structure
            current_round_post_followers={
                p["sender_id"]: (
                    list(self.network.followers(p["sender_id"])) if self.network else []
                )
                for p in self.current_round_posts_buffer
                if p["sender_id"] != self.rumor_spreader_id
            },
            current_round_post_following={
                p["sender_id"]: (
                    list(self.network.following(p["sender_id"])) if self.network else []
                )
                for p in self.current_round_posts_buffer
                if p["sender_id"] != self.rumor_spreader_id
            },
            # Ban agent quotas and usage
            current_round_ban_agent_usage=self.current_round_quota_usage["ban_agent"],
            current_round_ban_agent_quota=self.intervention_quotas[
                InterventionType.BAN_AGENT
            ]["per_round"],
            global_ban_agent_usage=self.global_quota_usage["ban_agent"],
            global_ban_agent_quota=self.intervention_quotas[InterventionType.BAN_AGENT][
                "global"
            ],
            # Persuade agent quotas and usage
            current_round_persuade_agent_usage=self.current_round_quota_usage[
                "persuade_agent"
            ],
            current_round_persuade_agent_quota=self.intervention_quotas[
                InterventionType.PERSUADE_AGENT
            ]["per_round"],
            global_persuade_agent_usage=self.global_quota_usage["persuade_agent"],
            global_persuade_agent_quota=self.intervention_quotas[
                InterventionType.PERSUADE_AGENT
            ]["global"],
            # Delete post quotas and usage
            current_round_delete_post_usage=self.current_round_quota_usage[
                "delete_post"
            ],
            current_round_delete_post_quota=self.intervention_quotas[
                InterventionType.DELETE_POST
            ]["per_round"],
            global_delete_post_usage=self.global_quota_usage["delete_post"],
            global_delete_post_quota=self.intervention_quotas[
                InterventionType.DELETE_POST
            ]["global"],
            # Remove follower quotas and usage
            current_round_remove_follower_usage=self.current_round_quota_usage[
                "remove_follower"
            ],
            current_round_remove_follower_quota=self.intervention_quotas[
                InterventionType.REMOVE_FOLLOWER
            ]["per_round"],
            global_remove_follower_usage=self.global_quota_usage["remove_follower"],
            global_remove_follower_quota=self.intervention_quotas[
                InterventionType.REMOVE_FOLLOWER
            ]["global"],
            # Current agent info for risk assessment
            current_agent_degree=(
                self.network.get_degree(agent_id) if self.network else 0
            ),
            current_agent_offense_summary=offense_summary,
            current_processing_agent_id=agent_id,
            current_agent_intervention_count=len(
                [
                    i
                    for i in self.all_historical_interventions_log
                    if i.get("target_agent_id") == agent_id
                ]
            ),
        )

        if intervention_count is not None:
            temporary_context.current_agent_intervention_count = intervention_count
        await prompt.format(context=temporary_context)  # type: ignore
        try:
            response = await asyncio.wait_for(
                self.llm.atext_request(prompt.to_dialog()),
                timeout=300,
            )
            response = response.strip()
            if "1" in response:
                return True
            elif "0" in response:
                return False
            else:
                print(
                    f"[LLM风险评估] Agent {agent_id} 的风险评估LLM输出 '{response}' 非预期格式 (应为1或0)。默认非高风险。"
                )
                return False
        except Exception as e:
            print(f"[LLM风险评估] Agent {agent_id} 的风险评估失败: {e}。默认非高风险。")
            return False
