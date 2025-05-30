"""
Survey for the Env Ambassador Challenge.
"""

import random
import uuid
from datetime import datetime
from typing import Any

import jsonc
from agentsociety.survey.models import Page, Question, QuestionType, Survey


def create_rumor_spread_survey(
    survey_rumors: list[str] = [
        "大棚西瓜价格在2026年12月10日只要每斤5角钱。",
        "村上春树获得了2026年诺贝尔文学奖。",
        "苏伊士运河于2026年12月13日发生了大拥堵。",
    ],
) -> Survey:
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
        在本次模拟中，你接收并处理了各种信息。
        请你根据你所了解的全部信息，评估以下言论的可信度
        {rumor}
        请以0-10的数字形式给出你的评估结果及简要理由，0表示完全不可信，10表示完全可信。
        请给出评分数字，数字可以不是整数，但需要在0-10的区间内。请以如下json格式输出：
        {{
            "score": <YOUR-SCORE>,
            "reason": <YOUR-REASON>
        }}
    """

    # Create questions for the survey
    questions = [
        Question(
            name=f"消息置信度调查#{i}",
            title=prompt_final_survey.format(rumor=rumor),
            type=QuestionType.RATING,
            min_rating=0,
            max_rating=10,
        )
        for i, rumor in enumerate(survey_rumors)
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

    return survey


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
            scores.append(10 * random.random())
    
    final_score = sum(scores) / len(scores) if scores else 5
    return scores, final_score


# Example usage
if __name__ == "__main__":
    survey = create_rumor_spread_survey()
    print(f"Created survey: {survey.title}")
    print(f"Survey ID: {survey.id}")
    print(f"Number of questions: {len(survey.pages[0].elements)}")

    # Example response scoring
    example_responses = {f"消息置信度-{i}": 3 + i / 10 for i in range(3)}

    scores = extract_survey_scores(list(example_responses.values()))
    print(f"Example scores: {scores}")
