import random
from collections import deque
from typing import Any, Optional

import numpy as np
from agentsociety.agent.distribution import Distribution, sample_field_value
from agentsociety.agent.memory_config_generator import MemoryT

pareto_param = 8
payment_max_skill_multiplier_base = 950
payment_max_skill_multiplier = float(payment_max_skill_multiplier_base)
pmsm = payment_max_skill_multiplier
pareto_samples = np.random.pareto(pareto_param, size=(1000, 10))
clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
sorted_clipped_skills = np.sort(clipped_skills, axis=1)
agent_skills = list(sorted_clipped_skills.mean(axis=0))

__all__ = [
    "memory_config_envcitizen",
]


def memory_config_envcitizen(
    distributions: dict[str, Distribution],
    class_config: Optional[dict[str, MemoryT]] = None,
) -> tuple[dict[str, MemoryT], dict[str, MemoryT], dict[str, Any]]:
    EXTRA_ATTRIBUTES = {
        "type": (str, "citizen"),
        # Needs Model
        "hunger_satisfaction": (float, 0.9, False),  # hunger satisfaction
        "energy_satisfaction": (float, 0.9, False),  # energy satisfaction
        "safety_satisfaction": (float, 0.4, False),  # safety satisfaction
        "social_satisfaction": (float, 0.6, False),  # social satisfaction
        "current_need": (str, "none", False),
        # Plan Behavior Model
        "current_plan": (dict, {}, False),
        "execution_context": (dict, {}, False),
        "plan_history": (list, [], False),
        # cognition
        "emotion": (
            dict,
            {
                "sadness": 5,
                "joy": 5,
                "fear": 5,
                "disgust": 5,
                "anger": 5,
                "surprise": 5,
            },
            False,
        ),
        "attitude": (dict, {}, True),
        "thought": (str, "Currently nothing good or bad is happening", True),
        "emotion_types": (str, "Relief", True),
        # economy
        "work_skill": (
            float,
            random.choice(agent_skills),
            True,
        ),  # work skill
        "tax_paid": (float, 0.0, False),  # tax paid
        "consumption_currency": (float, 0.0, False),  # consumption
        "goods_demand": (int, 0, False),
        "goods_consumption": (int, 0, False),
        "work_propensity": (float, 0.0, False),
        "consumption_propensity": (float, 0.0, False),
        "to_consumption_currency": (float, 0.0, False),
        "firm_id": (int, 0, False),
        "government_id": (
            int,
            0,
            False,
        ),
        "bank_id": (int, 0, False),
        "nbs_id": (int, 0, False),
        "dialog_queue": (deque(maxlen=3), [], False),
        "firm_forward": (int, 0, False),
        "bank_forward": (int, 0, False),
        "nbs_forward": (int, 0, False),
        "government_forward": (int, 0, False),
        "forward": (int, 0, False),
        "depression": (float, 0.0, False),
        "ubi_opinion": (list, [], False),
        "working_experience": (list, [], False),
        "work_hour_month": (float, 160, False),
        "work_hour_finish": (float, 0, False),
        # social
        "followers": (list, [], False),  # followers list
        "following": (list, [], False),  # following list
        "friends_info": (dict, {}, False),  # friends info
        "relationships": (dict, {}, False),  # relationship strength with each friend
        "relation_types": (dict, {}, False),
        "chat_histories": (dict, {}, False),  # all chat histories
        "interactions": (dict, {}, False),  # all interaction records
        # mobility
        "number_poi_visited": (int, 1, False),
        "location_knowledge": (dict, {}, False),  # location knowledge
        "message_propagation_preference": (
            str,
            "",
            False,
        ),
        "background_story": (
            str,
            "",
            False,
        ),
    }

    PROFILE = {
        "name": (str, sample_field_value(distributions, "name"), True),
        "gender": (str, sample_field_value(distributions, "gender"), True),
        "age": (int, sample_field_value(distributions, "age"), True),
        "education": (str, sample_field_value(distributions, "education"), True),
        "skill": (str, sample_field_value(distributions, "skill"), True),
        "occupation": (str, sample_field_value(distributions, "occupation"), True),
        "family_consumption": (
            str,
            sample_field_value(distributions, "family_consumption"),
            True,
        ),
        "consumption": (str, sample_field_value(distributions, "consumption"), True),
        "personality": (str, sample_field_value(distributions, "personality"), True),
        "income": (float, sample_field_value(distributions, "income"), True),
        "currency": (float, sample_field_value(distributions, "currency"), True),
        "residence": (str, sample_field_value(distributions, "residence"), True),
        "city": (str, sample_field_value(distributions, "city"), True),
        "race": (str, sample_field_value(distributions, "race"), True),
        "religion": (str, sample_field_value(distributions, "religion"), True),
        "marital_status": (
            str,
            sample_field_value(distributions, "marital_status"),
            True,
        ),
    }

    BASE = {
        "home": {
            "aoi_position": {"aoi_id": sample_field_value(distributions, "home_aoi_id")}
        },
        "work": {
            "aoi_position": {"aoi_id": sample_field_value(distributions, "work_aoi_id")}
        },
    }

    return EXTRA_ATTRIBUTES, PROFILE, BASE
