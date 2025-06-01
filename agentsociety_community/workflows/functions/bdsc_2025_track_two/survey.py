"""
Survey for the Env Ambassador Challenge.
"""

import random
import uuid
from datetime import datetime
from typing import Any

import jsonc
from agentsociety.survey.models import Page, Question, QuestionType, Survey


def create_rumor_spread_surveys(
    survey_rumors: list[str] = [
        "大棚西瓜价格在2026年12月10日只要每斤5角钱。",
    ],
) -> list[Survey]:
    """
    Creates a survey to assess urban residents' awareness of rumor spread.

    - **Description**:
        - Generates a 10-question survey focused on daily decisions related to rumor spread.
        - Options have clear distinctions in terms of rumor spread level.

    - **Returns**:
        - `survey` (Survey): A complete survey object ready for distribution.
    """
    survey_id = uuid.uuid4()

    prompt_final_survey = """       
        在社交网络平台上，你接收了各种信息。 请你根据你所预先了解到的信息，评估你刚刚听到的这个信息的可信度：{rumor}。\n请以0-10的数字形式给出你的评估结果，0代表完全不可信，10代表完全可信。\n请只给出数字，数字可以不是整数，但需要在0-10的区间内。不要输出任何其他内容。
    """

    # Create questions for the survey
    surveys = []
    for i, rumor in enumerate(survey_rumors):
        questions = [
            Question(
                name=f"消息置信度调查",
                title=prompt_final_survey.format(rumor=rumor),
                type=QuestionType.RATING,
                min_rating=0,
                max_rating=10,
            )
        ]

        # Create the page containing all questions
        page = Page(name="rumor_spread", elements=questions)

        # Create the complete survey
        survey = Survey(
            id=survey_id,
            title="谣言传播调查",
            description="本调查旨在评估城市居民在日常生活中的谣言传播意识和行为习惯。每个选项反映不同程度的谣言传播意识。",
            pages=[page],
            created_at=datetime.now(),
        )
        surveys.append(survey)

    return surveys


def extract_survey_scores(responses: list[Any]) -> tuple[list[float], float]:
    """
    Extracts the rumor spread scores from survey responses.

    - **Description**:
        - Lower scores indicate stronger rumor spread level.

    - **Args**:
        - `responses` (list[Any]): List of responses from the survey.

    - **Returns**:
        - `scores` (list[float]): List of scores from the survey.
    """
    scores = []

    for answer in responses:
        try:
            answer = jsonc.loads(answer)
            if isinstance(answer, dict):
                scores.append(float(answer["score"]))
            else:
                scores.append(float(answer))
        except Exception as e:
            pass
            # scores.append(10 * random.random())

    final_score = sum(scores) / len(scores) if scores else 5
    return scores, final_score
