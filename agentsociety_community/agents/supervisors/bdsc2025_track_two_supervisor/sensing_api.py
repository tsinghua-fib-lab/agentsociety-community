from __future__ import annotations

import re  # 用于关键词搜索
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

if TYPE_CHECKING:
    from supervisor import SupervisorBase


class InterventionType(str, Enum):
    DELETE_POST = "delete_post"  # 删除消息
    PERSUADE_AGENT = "persuade_agent"  # 劝导智能体
    REMOVE_FOLLOWER = "remove_follower"  # 移除关注
    BAN_AGENT = "ban_agent"  # 封禁智能体


class SensingAPI:
    def __init__(self, intervention_quotas: dict[InterventionType, dict[str, int]]):
        self.intervention_quotas = intervention_quotas
        self._supervisor = None

    def _set_supervisor(self, supervisor: "SupervisorBase"):
        self._supervisor = supervisor

    @property
    def supervisor(self) -> "SupervisorBase":
        assert self._supervisor is not None, "supervisor not set"
        return self._supervisor

    # --- 消息传播信息获取函数 --- #

    def get_all_historical_posts(self) -> list[dict[str, Any]]:
        """获取从比赛开始到当前轮次之前（包含当前轮已处理完的帖子）的所有公开帖子。"""
        # global_posts_history 已经包含了所有已处理的帖子
        return [p.copy() for p in self.supervisor.global_posts_history]  # type: ignore

    def get_posts_last_k_rounds(self, k: int) -> list[dict[str, Any]]:
        """获取最近 k 个已完成轮次的全部公开帖子。不包括当前正在进行的轮次。"""
        if k <= 0:
            return []
        current_round = (
            self.supervisor.get_current_round_number()  # type: ignore
        )  # 需要 Simulation 实现此方法
        # 我们只关心已完成的轮次，所以是从 current_round - 1 开始倒数 k 轮
        # 如果 current_round 是 1 (第一轮还没完成)，则不应返回任何东西
        if current_round <= 1:
            return []

        target_rounds = set(range(max(1, current_round - k), current_round))
        return [
            p.copy()
            for p in self.supervisor.global_posts_history  # type: ignore
            if p["round"] in target_rounds
        ]

    def get_posts_current_round(self) -> list[dict[str, Any]]:
        """获取本轮到目前为止产生的所有公开帖子（在监管者调用时，即预备帖子列表）。"""
        return [p.copy() for p in self.supervisor.current_round_posts_buffer]  # type: ignore

    def _filter_posts_by_receiver(
        self, posts: list[dict[str, Any]], agent_id: int
    ) -> list[dict[str, Any]]:
        """辅助函数：筛选出指定接收者的帖子"""
        received_posts = []
        for post in posts:
            # 关键点：是看 original_intended_receiver_ids 还是 actual_receiver_ids?
            # 感知接口通常用于监管决策，此时可能需要看到原始意图
            # 但如果是在轮次结束后分析实际传播，则看 actual
            # 根据描述 "作为接收者接收到的"，应理解为实际接收到的
            # 但在监管阶段，actual_receiver_ids 可能尚未最终确定（如果监管者可以修改它）
            # 为了简单和一致，我们先基于 original_intended_receiver_ids，后续可调整
            # **修正**：感知接口应反映干预前的状态，所以用 original_intended_receiver_ids
            # 如果监管者需要知道干预后的，可以再提供一个接口或参数
            # 或者，感知接口拿到的帖子数据本身就应该是"预备"状态
            # 您的流程是：1. 生成预备帖子 -> 2. 收集 -> 3. 调用监管（监管拿到的是预备帖子）
            # 所以，这里的 post['original_intended_receiver_ids'] 是合适的
            if agent_id in post.get("original_intended_receiver_ids", []):
                received_posts.append(post.copy())
        return received_posts

    def get_all_posts_received_by_agent(self, agent_id: int) -> list[dict[str, Any]]:
        """获取指定智能体在所有历史轮次中（含当前轮预备接收）作为接收者意图接收的所有公开帖子。"""
        all_posts = self.get_all_historical_posts() + self.get_posts_current_round()
        # 去重，因为 current_round_posts_buffer 的内容在轮次结束时也会加入 global_posts_history
        # 通过 post_id 去重
        unique_posts = {p["post_id"]: p for p in all_posts}.values()
        return self._filter_posts_by_receiver(list(unique_posts), agent_id)

    def get_posts_received_by_agent_last_k_rounds(
        self, agent_id: int, k: int
    ) -> list[dict[str, Any]]:
        """获取指定智能体在最近 k 个已完成轮次中作为接收者意图接收的所有公开帖子。"""
        posts_in_k_rounds = self.get_posts_last_k_rounds(k)
        return self._filter_posts_by_receiver(posts_in_k_rounds, agent_id)

    def get_posts_received_by_agent_current_round(
        self, agent_id: int
    ) -> list[dict[str, Any]]:
        """获取指定智能体在本轮作为接收者意图接收的所有公开帖子。"""
        current_posts = self.get_posts_current_round()
        return self._filter_posts_by_receiver(current_posts, agent_id)

    def _filter_posts_by_sender(
        self, posts: list[dict[str, Any]], agent_id: int
    ) -> list[dict[str, Any]]:
        """辅助函数：筛选出指定发送者的帖子"""
        return [p.copy() for p in posts if p.get("sender_id") == agent_id]

    def get_all_posts_sent_by_agent(self, agent_id: int) -> list[dict[str, Any]]:
        """获取指定智能体在所有历史轮次中（含当前轮预备发送）发送的所有公开帖子。"""
        all_posts = self.get_all_historical_posts() + self.get_posts_current_round()
        unique_posts = {p["post_id"]: p for p in all_posts}.values()
        return self._filter_posts_by_sender(list(unique_posts), agent_id)

    def get_posts_sent_by_agent_last_k_rounds(
        self, agent_id: int, k: int
    ) -> list[dict[str, Any]]:
        """获取指定智能体在最近 k 个已完成轮次中发送的所有公开帖子。"""
        posts_in_k_rounds = self.get_posts_last_k_rounds(k)
        return self._filter_posts_by_sender(posts_in_k_rounds, agent_id)

    def get_posts_sent_by_agent_current_round(
        self, agent_id: int
    ) -> list[dict[str, Any]]:
        """获取指定智能体在本轮到目前为止预备发送的所有公开帖子。"""
        current_posts = self.get_posts_current_round()
        return self._filter_posts_by_sender(current_posts, agent_id)

    def get_post_content_by_id(self, post_id: str) -> Optional[str]:
        """根据消息ID获取其具体内容（从全局历史和当前轮缓存中查找）。"""
        for post in self.supervisor.current_round_posts_buffer:  # type: ignore
            if post["post_id"] == post_id:
                return post["content"]
        for post in self.supervisor.global_posts_history:  # type: ignore
            if post["post_id"] == post_id:
                return post["content"]
        return None

    def get_posts_containing_keywords(
        self,
        keywords: list[str],
        search_scope: str = "current_round_all_senders",
        agent_id_context: Optional[int] = None,
        last_k_rounds_context: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """在公域消息中搜索包含指定任意一个关键词的消息。"""
        if not keywords:
            return []

        posts_to_search: list[dict[str, Any]] = []
        current_sim_round = self.supervisor.get_current_round_number()  # type: ignore
        effective_last_k = (
            last_k_rounds_context
            if last_k_rounds_context is not None and last_k_rounds_context > 0
            else current_sim_round
        )

        if search_scope == "current_round_all_senders":
            posts_to_search = self.get_posts_current_round()
        elif search_scope == "current_round_sent_by_agent":
            if agent_id_context is None:
                raise ValueError(
                    "agent_id_context is required for current_round_sent_by_agent"
                )
            posts_to_search = self.get_posts_sent_by_agent_current_round(
                agent_id_context
            )
        elif search_scope == "current_round_received_by_agent":
            if agent_id_context is None:
                raise ValueError(
                    "agent_id_context is required for current_round_received_by_agent"
                )
            posts_to_search = self.get_posts_received_by_agent_current_round(
                agent_id_context
            )
        elif search_scope == "historical_sent_by_agent":
            if agent_id_context is None:
                raise ValueError(
                    "agent_id_context is required for historical_sent_by_agent"
                )
            base_posts = self.get_all_posts_sent_by_agent(agent_id_context)
            target_rounds = set(
                range(
                    max(1, current_sim_round - effective_last_k + 1),
                    current_sim_round + 1,
                )
            )
            posts_to_search = [p for p in base_posts if p["round"] in target_rounds]
        elif search_scope == "historical_received_by_agent":
            if agent_id_context is None:
                raise ValueError(
                    "agent_id_context is required for historical_received_by_agent"
                )
            base_posts = self.get_all_posts_received_by_agent(agent_id_context)
            target_rounds = set(
                range(
                    max(1, current_sim_round - effective_last_k + 1),
                    current_sim_round + 1,
                )
            )
            posts_to_search = [p for p in base_posts if p["round"] in target_rounds]
        elif search_scope == "all_historical":
            base_posts = (
                self.get_all_historical_posts() + self.get_posts_current_round()
            )
            unique_posts = list({p["post_id"]: p for p in base_posts}.values())
            target_rounds = set(
                range(
                    max(1, current_sim_round - effective_last_k + 1),
                    current_sim_round + 1,
                )
            )
            posts_to_search = [p for p in unique_posts if p["round"] in target_rounds]
        else:
            raise ValueError(f"未知的 search_scope: {search_scope}")

        # 正则表达式匹配任何一个关键词 (忽略大小写)
        # pattern = "|".join(map(re.escape, keywords))
        # compiled_regex = re.compile(pattern, re.IGNORECASE)
        # matched_posts = [p.copy() for p in posts_to_search if compiled_regex.search(p.get("content", ""))]

        # 简单字符串匹配（任意一个关键词，忽略大小写）
        lower_keywords = [k.lower() for k in keywords]
        matched_posts = []
        for p in posts_to_search:
            content_lower = p.get("content", "").lower()
            if any(kw in content_lower for kw in lower_keywords):
                matched_posts.append(p.copy())

        return matched_posts

    # --- 智能体状态与干预历史类函数 (12-14, 16-19) --- #

    def get_agent_offending_posts_current_round(
        self, agent_id: int
    ) -> list[dict[str, Any]]:
        """获取指定智能体在本轮发送的、被监管判定为违规的帖子。"""
        agent_posts_current_round = self.get_posts_sent_by_agent_current_round(agent_id)
        offending_posts = []
        for post in agent_posts_current_round:
            detection = post.get("detection_results", {})
            if detection.get("is_violating", False):
                offending_posts.append(
                    {
                        "post_id": post["post_id"],
                        "content": post["content"],
                        "detection_method": detection.get("method", "unknown"),
                    }
                )
        return offending_posts

    def get_agent_historical_offense_rounds(
        self, agent_id: int
    ) -> list[dict[str, Union[int, str]]]:
        """获取指定智能体所有发送过违规帖子的轮次及首个违规帖子内容摘要。"""
        # 包含当前轮的检测结果
        all_sent_posts = self.get_all_posts_sent_by_agent(agent_id)
        offense_rounds_data = {}
        for post in all_sent_posts:
            detection = post.get("detection_results", {})
            if detection.get("is_violating", False):
                round_num = post["round"]
                if round_num not in offense_rounds_data:
                    offense_rounds_data[round_num] = (
                        post["content"][:50] + "..."
                    )  # 只取摘要

        # 按轮次排序
        sorted_offenses = []
        for r in sorted(offense_rounds_data.keys()):
            sorted_offenses.append({str(r): offense_rounds_data[r]})
        return sorted_offenses

    def get_agent_intervention_history(self, agent_id: int) -> list[dict[str, Any]]:
        """获取指定智能体受到的所有干预措施的历史记录。"""
        agent = self.supervisor.agent_map.get(agent_id)  # type: ignore
        if agent:
            return [
                h.copy() for h in agent.intervention_history
            ]  # 返回拷贝以防外部修改
        return []

    # 接口15. get_agent_public_post_count_by_type 在描述中缺失，跳过

    def count_posts_between_agents(
        self,
        sender_id: int,
        receiver_id: int,
        count_scope: str = "all_historical",
        last_k_rounds: Optional[int] = None,
    ) -> int:
        """统计在公域网络中，一个智能体的帖子被另一个智能体看到的次数（基于原始意图）。"""
        posts_to_consider: list[dict[str, Any]] = []
        current_sim_round = self.supervisor.get_current_round_number()  # type: ignore
        effective_last_k = (
            last_k_rounds
            if last_k_rounds is not None and last_k_rounds > 0
            else current_sim_round
        )

        if count_scope == "all_historical":
            posts_to_consider = self.get_all_posts_sent_by_agent(sender_id)
        elif count_scope == "current_round":
            posts_to_consider = self.get_posts_sent_by_agent_current_round(sender_id)
        elif count_scope == "last_k_rounds":
            target_rounds = set(
                range(
                    max(1, current_sim_round - effective_last_k + 1),
                    current_sim_round + 1,
                )
            )
            all_sender_posts = self.get_all_posts_sent_by_agent(sender_id)
            posts_to_consider = [
                p for p in all_sender_posts if p["round"] in target_rounds
            ]
        else:
            raise ValueError(f"未知的 count_scope: {count_scope}")

        count = 0
        for post in posts_to_consider:
            # 这里我们检查原始意图的接收者中是否有目标接收者
            if receiver_id in post.get("original_intended_receiver_ids", []):
                count += 1
        return count

    def get_agent_historical_offense_summary(self, agent_id: int) -> dict[str, int]:
        """获取指定智能体历史上的"违规行为"概要统计。"""
        summary = {
            "total_rumors_sent_by_llm_detection": 0,
            "total_rumors_sent_by_keyword_detection": 0,
            "times_communicated_by_regulator": 0,
            "times_its_messages_blocked": 0,
        }
        # 1. 统计违规帖子发送 (依赖帖子的detection_results)
        all_sent_posts = self.get_all_posts_sent_by_agent(agent_id)
        for post in all_sent_posts:
            detection = post.get("detection_results", {})
            if detection.get("is_violating", False):
                if detection.get("method") == "llm":
                    summary["total_rumors_sent_by_llm_detection"] += 1
                elif detection.get("method") == "keyword":
                    summary["total_rumors_sent_by_keyword_detection"] += 1

        # 2. 统计被干预次数 (依赖智能体的intervention_history)
        intervention_history = self.get_agent_intervention_history(agent_id)
        for record in intervention_history:
            if record.get("intervention_type") == "persuasion_received":
                summary["times_communicated_by_regulator"] += 1
            elif record.get("intervention_type") == "post_deleted":  # 假设删帖也算
                summary["times_its_messages_blocked"] += 1
            # 注意：更精确的 times_its_messages_blocked 应直接统计被is_blocked_by_regulator的帖子
            # 但此处先按被干预历史中的 post_deleted 类型统计
        return summary

    def get_public_edge_rumor_traffic(self, agent_id_1: int, agent_id_2: int) -> int:
        """获取某条公域网络边上累计传播的、被判定为违规消息的数量（基于原始意图）。"""
        # agent_id_1 (source) -> agent_id_2 (target)
        posts_from_sender = self.get_all_posts_sent_by_agent(agent_id_1)
        rumor_traffic_count = 0
        for post in posts_from_sender:
            detection = post.get("detection_results", {})
            # 检查是否违规，并且接收者中包含 agent_id_2
            if detection.get("is_violating", False) and agent_id_2 in post.get(
                "original_intended_receiver_ids", []
            ):
                rumor_traffic_count += 1
        return rumor_traffic_count

    def get_rumor_message_ids_current_round(self) -> list[str]:
        """获取本轮所有被监管者判定为谣言的消息ID列表。"""
        ids = []
        for post in self.get_posts_current_round():  # 本轮预备帖子
            detection = post.get("detection_results", {})
            if detection.get("is_violating", False):
                ids.append(post["post_id"])
        return ids

    # --- 网络结构信息类函数 (20-24) --- #
    def get_public_network_structure(self) -> dict[str, list[Any]]:
        """获取当前公域网络的结构。边列表可能包含重复（如果原始数据如此）。"""
        return self.supervisor.network.get_network_structure()  # type: ignore

    def get_public_node_degree(self, agent_id: int) -> Tuple[int, int]:
        """获取指定智能体在当前网络中的入度 (多少人关注TA) 和出度 (TA关注多少人)。"""
        # 入度：有多少人关注 agent_id -> agent_id 是 target
        in_degree = len(self.supervisor.network.followers(agent_id))  # type: ignore
        # 出度：agent_id 关注了多少人 -> agent_id 是 source
        out_degree = len(self.supervisor.network.following(agent_id))  # type: ignore
        return in_degree, out_degree

    def get_top_degree_nodes(self, top_k: int, degree_type: str = "total") -> list[int]:
        """获取当前网络中指定类型度排名前 top_k 的智能体ID列表。"""
        if top_k <= 0:
            return []
        degrees_map = {}
        for node_id in self.supervisor.agent_map.keys():  # type: ignore
            in_degree, out_degree = self.get_public_node_degree(node_id)
            if degree_type == "in":
                degrees_map[node_id] = in_degree
            elif degree_type == "out":
                degrees_map[node_id] = out_degree
            elif degree_type == "total":
                degrees_map[node_id] = in_degree + out_degree
            else:
                raise ValueError(
                    f"未知的 degree_type: {degree_type}. 可选 'in', 'out', 'total'."
                )

        sorted_nodes = sorted(
            degrees_map.items(), key=lambda item: item[1], reverse=True
        )
        return [node_id for node_id, degree in sorted_nodes[:top_k]]

    def get_public_node_at_degree_rank(
        self, rank: int, degree_type: str = "total"
    ) -> Optional[int]:
        """获取当前公域网络中指定类型度排名第 rank 位的智能体ID。排名从1开始。"""
        if rank <= 0:
            return None
        top_nodes = self.get_top_degree_nodes(rank, degree_type)
        return top_nodes[rank - 1] if len(top_nodes) >= rank else None

    def get_public_neighbors(self, agent_id: int) -> list[int]:
        """获取指定智能体在当前公域网络中的所有邻居（TA关注的 + 关注TA的）。"""
        if agent_id not in self.supervisor.agent_map:  # type: ignore
            return []
        followers = self.supervisor.network.followers(agent_id)  # type: ignore
        following = self.supervisor.network.following(agent_id)  # type: ignore
        return list(followers.union(following))

    # --- 监管者自身状态与干预配额类函数 (25-28) --- #
    def get_remaining_intervention_quotas(self) -> dict[str, dict[str, int]]:
        """获取当前监管智能体剩余的各类干预措施的配额。"""
        remaining_quotas = {}
        for type, limits in self.intervention_quotas.items():
            used_stats = self.supervisor.intervention_stats.get(  # type: ignore
                type, {"per_round_used": 0, "global_used": 0}
            )
            remaining_round = (
                limits["per_round"] - used_stats["per_round_used"]
                if limits["per_round"] != -1
                else float("inf")
            )
            remaining_global = (
                limits["global"] - used_stats["global_used"]
                if limits["global"] != -1
                else float("inf")
            )
            remaining_quotas[type] = {
                "round_remaining": remaining_round,
                "global_remaining": remaining_global,
                "round_limit": limits["per_round"],
                "global_limit": limits["global"],
            }
        return remaining_quotas

    def get_globally_blocked_elements(self) -> dict[str, list[Any]]:
        """获取已经被永久封禁的智能体和被永久移除的关注关系。"""
        return {
            "blocked_agents": sorted(list(self.supervisor.banned_agent_ids)),  # type: ignore
            "cut_edges": sorted(
                [list(edge) for edge in self.supervisor.globally_removed_edges]  # type: ignore
            ),
        }

    def get_interventions_by_me_this_round(self) -> list[dict[str, Any]]:
        """获取当前监管智能体在本轮已经执行的干预操作列表。"""
        return [
            intervention.copy()
            for intervention in self.supervisor.current_round_interventions  # type: ignore
        ]

    def get_interventions_by_me(self) -> list[dict[str, Any]]:
        """获取当前监管智能体全部（历史）已经执行的干预操作列表。"""
        return [
            intervention.copy()
            for intervention in self.supervisor.all_historical_interventions_log  # type: ignore
        ]

    # --- 平台与比赛信息类函数 (29-31) --- #
    def get_current_round_number(self) -> int:
        """获取当前的比赛轮次。"""
        return self.supervisor.get_current_round_number()  # type: ignore

    def get_total_simulation_rounds(self) -> int:
        """获取本次模拟比赛总共的轮次数。"""
        return self.supervisor.get_total_simulation_rounds()  # type: ignore

    def get_rumor_topic_description(self) -> list[str]:
        """获取当前比赛设定的核心谣言主题的文字描述（问卷用）。"""
        return self.supervisor.rumor_topic_description  # type: ignore

    # ... (其他方法保持不变) ...
