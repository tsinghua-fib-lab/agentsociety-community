import asyncio
import random
import time
from abc import abstractmethod
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
    InterventionType.DELETE_POST: {"per_round": 20, "global": -1},  # -1 表示无全局限制
    InterventionType.PERSUADE_AGENT: {"per_round": 20, "global": -1},
    InterventionType.REMOVE_FOLLOWER: {"per_round": -1, "global": 100},
    InterventionType.BAN_AGENT: {"per_round": -1, "global": 20},
}


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


class BDSC2025SupervisorBase(SupervisorBase):
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

        # # ATTENTION: dump干预log
        # with open(
        #     f"all_historical_interventions_log.json",
        #     "w",
        # ) as f:
        #     jsonc.dump(self.all_historical_interventions_log, f, ensure_ascii=False)

        # with open(
        #     f"all_historical_responses.json",
        #     "w",
        # ) as f:
        #     jsonc.dump(self.all_responses, f, ensure_ascii=False)

        # 更新时间记录
        self.time_record[self.current_round_number]["end_time"] = time.time()
        # 更新轮次计数
        self.current_round_number += 1
        self.global_posts_history.extend(self.current_round_posts_buffer)
        # with open(
        #     f"time_record.json",
        #     "w",
        # ) as f:
        #     jsonc.dump(self.time_record, f)
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

    @abstractmethod
    async def interventions(
        self,
    ) -> None:
        """
        检测并执行干预措施
        """
        raise NotImplementedError("Interventions are not implemented in the base class")

    def ban_agent_intervention(self, agent_id: int, reason: str = "") -> bool:
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

    def delete_post_intervention(self, post_id: str, reason: str = "") -> bool:
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
