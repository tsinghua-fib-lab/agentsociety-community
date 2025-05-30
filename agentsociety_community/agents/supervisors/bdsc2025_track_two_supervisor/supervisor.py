import asyncio
import random
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Set, Tuple, cast

import jsonc
import pandas as pd
from agentsociety.agent import (AgentToolbox, Block, StatusAttribute,
                                SupervisorBase)
from agentsociety.agent.prompt import FormatPrompt
from agentsociety.llm import LLM
from agentsociety.memory import Memory
from agentsociety.utils.parsers import JsonDictParser
from openai import OpenAIError

from .sensing_api import InterventionType, SensingAPI
from .sharing_params import SupervisorConfig, SupervisorContext

DEFAULT_INTERVENTION_QUOTAS: dict[InterventionType, dict[str, int]] = {
    InterventionType.DELETE_POST: {"per_round": 10, "global": -1},  # -1 表示无全局限制
    InterventionType.PERSUADE_AGENT: {"per_round": 10, "global": -1},
    InterventionType.REMOVE_FOLLOWER: {"per_round": -1, "global": 100},
    InterventionType.BAN_AGENT: {"per_round": -1, "global": 20},
}


class RelationNode:
    def __init__(self, id: int):
        self.id = id
        self.following = set()
        self.followers = set()


class RelationNetwork:
    def __init__(self, social_network: dict[int, dict[str, list[int]]]):
        self.nodes: dict[int, RelationNode] = {}
        self.degrees: dict[int, int] = {}  # 添加度数统计

        # 遍历社交网络字典
        for agent_id, connections in social_network.items():
            # 初始化节点
            if agent_id not in self.nodes:
                self.nodes[agent_id] = RelationNode(agent_id)
                self.degrees[agent_id] = 0

            # 处理关注者
            for follower_id in connections.get("followers", []):
                if follower_id not in self.nodes:
                    self.nodes[follower_id] = RelationNode(follower_id)
                    self.degrees[follower_id] = 0
                self.nodes[follower_id].following.add(agent_id)
                self.nodes[agent_id].followers.add(follower_id)
                self.degrees[follower_id] += 1
                self.degrees[agent_id] += 1

            # 处理关注的人
            for following_id in connections.get("following", []):
                if following_id not in self.nodes:
                    self.nodes[following_id] = RelationNode(following_id)
                    self.degrees[following_id] = 0
                self.nodes[agent_id].following.add(following_id)
                self.nodes[following_id].followers.add(agent_id)
                self.degrees[agent_id] += 1
                self.degrees[following_id] += 1

    def following(self, node_id: int) -> Set[int]:
        if node_id not in self.nodes:
            return set()
        return self.nodes[node_id].following

    def followers(self, node_id: int) -> Set[int]:
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
    ParamsType = SupervisorConfig
    BlockOutputType = Any
    Context = SupervisorContext
    StatusAttributes = [
        # Needs Model
        StatusAttribute(
            name="social_network",
            type=dict,
            default={},
            description="agent's social network",
        ),
    ]

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
        self._current_validation_dict: dict[tuple[int, int, str], bool] = {}
        self._current_blocked_agent_ids: list[int] = []
        self._current_blocked_social_edges: list[tuple[int, int]] = []
        self._current_persuasion_messages: list[dict[str, Any]] = []

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

    async def supervisor_func(
        self, current_round_messages: list[tuple[int, int, str]]
    ) -> tuple[
        dict[tuple[int, int, str], bool],
        list[int],
        list[tuple[int, int]],
        list[dict[str, Any]],
    ]:
        """
        处理当前轮次的消息，进行验证和干预

        Args:
            current_round_messages: 当前轮次的消息列表，每个元素为 (sender_id, receiver_id, content) 的元组
            llm: LLM实例，用于内容验证

        Returns:
            validation_dict: 消息验证结果字典，key为消息元组，value为是否通过验证
            blocked_agent_ids: 被封禁的智能体ID列表
            blocked_social_edges: 被阻止的社交边列表
            persuasion_messages: 劝导消息列表
        """
        # 初始化网络结构
        if self.network is None:
            # agent id -> following & followers
            social_network: dict[int, dict[str, list[int]]] = (
                await self.memory.status.get("social_network", {})
            )
            self.network = RelationNetwork(social_network)
        assert self.network is not None, "Network is not initialized"
        # 清空当前轮次的缓冲区和干预记录
        added_sender_and_msg: set[tuple[int, str]] = set()
        identifier_to_post: dict[tuple[int, str], dict[str, Any]] = {}
        for sender_id, receiver_id, msg in current_round_messages:
            content = jsonc.loads(msg)["content"]
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
        self._current_validation_dict = {}
        self._current_blocked_agent_ids = []
        self._current_blocked_social_edges = []
        self._current_persuasion_messages = []

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
        # high score posts
        self.context.current_round_high_score_posts = []
        # network structure
        self.context.current_round_post_followers = {  # type: ignore
            post["sender_id"]: self.network.followers(post["sender_id"])
            for post in self.current_round_posts_buffer
        }
        self.context.current_round_post_following = {  # type: ignore
            post["sender_id"]: self.network.following(post["sender_id"])
            for post in self.current_round_posts_buffer
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

        # 更新轮次计数
        self.current_round_number += 1
        self.global_posts_history.extend(self.current_round_posts_buffer)

        return (
            self._current_validation_dict,
            self._current_blocked_agent_ids,
            self._current_blocked_social_edges,
            self._current_persuasion_messages,
        )

    async def interventions(
        self,
    ) -> None:
        """
        检测并执行干预措施
        """
        # 1. 检测阶段 - 检查当前轮次的所有帖子
        all_scores: list[float] = []

        async def get_score(post: dict[str, Any]) -> float:
            temporary_context = SupervisorContext()
            temporary_context.current_processing_message = post["content"]
            temporary_context.current_processing_message_sender_id = post["sender_id"]
            temporary_context.current_processing_message_receiver_ids = post[
                "original_intended_receiver_ids"
            ]
            preprocess_prompt = FormatPrompt(
                template=self.params.preprocess_message_prompt,
            )
            try:
                await preprocess_prompt.format(
                    context=temporary_context,  # type: ignore
                )
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        preprocess_prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                response = jsonc.loads(response)
                if isinstance(response, dict):
                    score = float(response["score"])
                else:
                    score = float(response)
            except Exception as e:
                score = 5 * random.random()
            print(f"Post {post['post_id']} score: {score}")
            return score

        # for post in self.sensing_api.get_posts_current_round():
        #     # set context
        #     self.context.current_processing_message = post["content"]
        #     self.context.current_processing_message_sender_id = post["sender_id"]
        #     self.context.current_processing_message_receiver_ids = post[
        #         "original_intended_receiver_ids"
        #     ]
        #     try:
        #         preprocess_prompt = FormatPrompt(
        #             template=self.params.preprocess_message_prompt,
        #         )
        #         await preprocess_prompt.format(
        #             context=self.context,  # type: ignore
        #         )
        #         response = await asyncio.wait_for(
        #             self.llm.atext_request(
        #                 preprocess_prompt.to_dialog(),
        #             ),
        #             timeout=300,
        #         )
        #         response = jsonc.loads(response)
        #         if isinstance(response, dict):
        #             score = float(response["score"])
        #         else:
        #             score = float(response)
        #     except Exception as e:
        #         score = 5 * random.random()
        #     all_scores.append(score)
        print(f"Processing {len(self.current_round_posts_buffer)} posts")
        tasks = [get_score(post) for post in self.current_round_posts_buffer]
        all_scores = await asyncio.gather(*tasks)
        # 2. 干预阶段 - 根据得分选出最高的max_process_message_per_round进行干预
        high_score_post_and_scores = sorted(
            zip(
                self.current_round_posts_buffer,
                all_scores,
            ),
            key=lambda x: x[1],
            reverse=True,
        )[: self.max_process_message_per_round]
        high_score_posts = [
            {
                "sender_id": post["sender_id"],
                "post_id": post["post_id"],
                "content": post["content"],
                "original_intended_receiver_ids": post[
                    "original_intended_receiver_ids"
                ],
            }
            for post, _ in high_score_post_and_scores
        ]
        # 2.1 删帖 (delete_post)
        # set high score posts
        self.context.current_round_high_score_posts = deepcopy(high_score_posts)
        for _ in range(self.max_retry_times):
            try:
                delete_post_prompt = FormatPrompt(
                    template=self.params.delete_post_prompt,
                )
                await delete_post_prompt.format(context=self.context)  # type: ignore
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        delete_post_prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                self.all_responses.append(f"Delete post response: {response}")
                print(f"Delete post response: {response}")
                response = JsonDictParser().parse(response)
                to_delete_post_ids = []
                for string_id in response["to_delete_post_ids"]:
                    to_delete_post_ids.append(str(string_id))
                print(f"To delete post ids: {to_delete_post_ids}")
                for post_id in to_delete_post_ids:
                    self.delete_post_intervention(post_id, "低分帖子")
                break
            except OpenAIError as e:
                self.context.current_round_high_score_posts = (
                    self.context.current_round_high_score_posts[
                        : -self.messages_shorten_length
                    ]
                )
            except Exception as e:
                print(f"Delete post error: {e}")
        # 2.2 劝说 (persuade_agent)
        # set high score posts
        self.context.current_round_high_score_posts = deepcopy(high_score_posts)
        for _ in range(self.max_retry_times):
            try:
                persuade_agent_prompt = FormatPrompt(
                    template=self.params.persuade_agent_prompt,
                )
                await persuade_agent_prompt.format(context=self.context)  # type: ignore
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        persuade_agent_prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                self.all_responses.append(f"Persuade agent response: {response}")
                print(f"Persuade agent response: {response}")
                response = JsonDictParser().parse(response)
                for agent_id, message in response.items():
                    self.persuade_agent_intervention(int(agent_id), message)
                break
            except OpenAIError as e:
                self.context.current_round_high_score_posts = (
                    self.context.current_round_high_score_posts[
                        : -self.messages_shorten_length
                    ]
                )
            except Exception as e:
                print(f"Persuade agent error: {e}")
        # 2.3 移除关注者 (remove_follower)
        # set high score posts
        self.context.current_round_high_score_posts = deepcopy(high_score_posts)
        for _ in range(self.max_retry_times):
            try:
                remove_follower_prompt = FormatPrompt(
                    template=self.params.remove_follower_prompt,
                )
                await remove_follower_prompt.format(context=self.context)  # type: ignore
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        remove_follower_prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                self.all_responses.append(f"Remove follower response: {response}")
                print(f"Remove follower response: {response}")
                response = JsonDictParser().parse(response)
                for follower_to_remove_id, following_agent_id in response.items():
                    self.remove_follower_intervention(
                        int(follower_to_remove_id), int(following_agent_id)
                    )
                break
            except OpenAIError as e:
                self.context.current_round_high_score_posts = (
                    self.context.current_round_high_score_posts[
                        : -self.messages_shorten_length
                    ]
                )
            except Exception as e:
                print(f"Remove follower error: {e}")
        # 2.4 封禁 (ban_agent)
        # set high score posts
        self.context.current_round_high_score_posts = deepcopy(high_score_posts)
        for _ in range(self.max_retry_times):
            try:
                ban_agent_prompt = FormatPrompt(
                    template=self.params.ban_agent_prompt,
                )
                await ban_agent_prompt.format(context=self.context)  # type: ignore
                response = await asyncio.wait_for(
                    self.llm.atext_request(
                        ban_agent_prompt.to_dialog(),
                    ),
                    timeout=300,
                )
                self.all_responses.append(f"Ban agent response: {response}")
                print(f"Ban agent response: {response}")
                response = JsonDictParser().parse(response)
                for string_id in response["to_ban_agent_ids"]:
                    if string_id.isdigit():
                        self.ban_agent_intervention(int(string_id), "低分帖子")
                    else:
                        if string_id.startswith("agent_id_"):
                            string_id = string_id[9:]
                            if string_id.isdigit():
                                self.ban_agent_intervention(int(string_id), "低分帖子")
                            else:
                                print(f"Invalid agent id: {string_id}")
                break
            except OpenAIError as e:
                self.context.current_round_high_score_posts = (
                    self.context.current_round_high_score_posts[
                        : -self.messages_shorten_length
                    ]
                )
            except Exception as e:
                print(f"Ban agent error: {e}")

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
        self._current_persuasion_messages.append(
            {
                "agent_id": agent_id,
                "message": jsonc.dumps({"type": "agent_banned"}, default=str),
            }
        )

        # 更新所有涉及该智能体的消息的验证结果
        for (
            sender_id,
            receiver_id,
            content,
        ), is_valid in self._current_validation_dict.items():
            if sender_id == agent_id or receiver_id == agent_id:
                self._current_validation_dict[(sender_id, receiver_id, content)] = False

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

        self._current_persuasion_messages.append(
            {
                "agent_id": target_agent_id,
                "message": jsonc.dumps(
                    {
                        "type": "persuasion",
                        "message": message,
                        "round": self.current_round_number,
                    },
                    default=str,
                ),
            }
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
        self._current_persuasion_messages.append(
            {
                "agent_id": post["sender_id"],
                "message": jsonc.dumps(
                    {"type": "post_deleted", "post_id": post_id}, default=str
                ),
            }
        )

        # 更新消息的验证结果
        for (
            sender_id,
            receiver_id,
            content,
        ), is_valid in list(self._current_validation_dict.items()):
            if (
                sender_id == post["sender_id"]
                and receiver_id in post["original_intended_receiver_ids"]
                and content == post["content"]
            ):
                self._current_validation_dict[(sender_id, receiver_id, content)] = False

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
        for (
            sender_id,
            receiver_id,
            content,
        ), is_valid in self._current_validation_dict.items():
            if sender_id == following_agent_id and receiver_id == follower_to_remove_id:
                self._current_validation_dict[(sender_id, receiver_id, content)] = False

        # 添加message给智能体 移除对应的follower和following
        self._current_persuasion_messages.append(
            {
                "agent_id": following_agent_id,
                "message": jsonc.dumps(
                    {
                        "type": "remove-follower",
                        "to_remove_id": follower_to_remove_id,
                    },
                    default=str,
                ),
            }
        )
        self._current_persuasion_messages.append(
            {
                "agent_id": follower_to_remove_id,
                "message": jsonc.dumps(
                    {
                        "type": "remove-following",
                        "to_remove_id": following_agent_id,
                    },
                    default=str,
                ),
            }
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


    async def forward(self):
        pass
