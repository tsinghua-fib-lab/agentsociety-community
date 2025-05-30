"""
Survey for the Env Ambassador Challenge.
"""

import json
from typing import Any
import uuid
from datetime import datetime
from agentsociety.survey.models import Survey, Page, Question, QuestionType
import jsonc

def create_low_carbon_awareness_survey() -> Survey:
    """
    Creates a survey to assess urban residents' awareness of low-carbon lifestyles.
    
    - **Description**:
        - Generates a 10-question survey focused on daily decisions related to carbon footprint.
        - Each question follows the [Profile]-[Environment]-[Goal] format.
        - Options have clear distinctions in terms of environmental impact.
    
    - **Returns**:
        - `survey` (Survey): A complete survey object ready for distribution.
    """
    survey_id = uuid.uuid4()
    
    # Create questions for the survey
    questions = [
        Question(
            name="transportation_short_distance",
            title="你经济状况良好，有私家车；现在要前往距离你当前位置3km的一家餐厅吃饭；你选择的出行交通方式是？",
            type=QuestionType.RADIO,
            choices=[
                "A. 步行",
                "B. 骑自行车或共享单车",
                "C. 乘坐公共交通",
                "D. 开私家车"
            ]
        ),
        Question(
            name="electricity_usage",
            title="你独自在家工作；夏天室外温度35度；关于室内降温，你会选择：",
            type=QuestionType.RADIO,
            choices=[
                "A. 拉上窗帘并使用电风扇",
                "B. 将空调温度设定在26-28度并定时关闭",
                "C. 全天开启空调保持室内舒适",
                "D. 将空调温度设定在20度以下确保凉爽"
            ]
        ),
        Question(
            name="shopping_bags",
            title="你计划去超市购物；关于购物袋的使用，你通常会：",
            type=QuestionType.RADIO,
            choices=[
                "A. 自带可重复使用的环保购物袋",
                "B. 尽量少用袋子，必要时使用超市提供的袋子",
                "C. 根据购物量选择合适数量的超市提供的袋子",
                "D. 为方便整理，每类商品使用单独的袋子"
            ]
        ),
        Question(
            name="diet_choices",
            title="你在选择日常饮食时，最符合你习惯的描述是：",
            type=QuestionType.RADIO,
            choices=[
                "A. 主要选择当季蔬果，很少食用肉类",
                "B. 注重饮食多样性，但会控制肉类摄入量",
                "C. 喜欢肉类但也会搭配一些蔬菜",
                "D. 偏好高蛋白饮食，肉类是主要食物来源"
            ]
        ),
        Question(
            name="water_usage",
            title="关于日常用水习惯，下列哪项最符合你的行为：",
            type=QuestionType.RADIO,
            choices=[
                "A. 收集洗菜水浇花，洗澡时间控制在5分钟内",
                "B. 注意随手关水龙头，洗澡时间控制在10分钟左右",
                "C. 偶尔会忘记关紧水龙头，喜欢舒适的淋浴体验",
                "D. 习惯放满浴缸泡澡，认为水费不是主要开支"
            ]
        ),
        Question(
            name="waste_sorting",
            title="当你处理家庭垃圾时，你通常会：",
            type=QuestionType.RADIO,
            choices=[
                "A. 严格按照可回收、厨余、有害和其他垃圾进行分类",
                "B. 基本将垃圾分为可回收和不可回收两大类",
                "C. 知道应该分类，但觉得太麻烦经常不执行",
                "D. 认为垃圾最终都会被处理，分不分类影响不大"
            ]
        ),
        Question(
            name="commuting",
            title="你在市区工作，住所距离办公室6公里；你日常通勤的首选方式是：",
            type=QuestionType.RADIO,
            choices=[
                "A. 骑自行车或电动自行车",
                "B. 乘坐公共交通工具",
                "C. 使用网约车或出租车",
                "D. 开私家车，享受便捷和私密空间"
            ]
        ),
        Question(
            name="electronics_usage",
            title="关于家中电器的使用习惯，你通常会：",
            type=QuestionType.RADIO,
            choices=[
                "A. 不用时关闭并拔掉插头，购买节能电器",
                "B. 使用智能插座管理待机电器，减少能耗",
                "C. 长时间不用的电器会关闭，但保持插电状态",
                "D. 为了方便随时使用，大多数电器保持待机状态"
            ]
        ),
        Question(
            name="leisure_activity",
            title="当你计划周末休闲活动时，你最倾向于：",
            type=QuestionType.RADIO,
            choices=[
                "A. 在家附近的公园或社区活动中心参与活动",
                "B. 乘坐公共交通去市内的文化场所或商场",
                "C. 驾车去郊外景点游玩一天",
                "D. 计划短途飞行或长途自驾出游"
            ]
        ),
        Question(
            name="consumption_habits",
            title="关于个人消费习惯，下列描述中最符合你的是：",
            type=QuestionType.RADIO,
            choices=[
                "A. 优先考虑物品的耐用性，尽量修复旧物而非购买新品",
                "B. 购物前会考虑是否真正需要，偶尔会为品质付出更多",
                "C. 跟随流行趋势更新物品，但会在促销时购买",
                "D. 喜欢拥有最新产品，经常更换手机、服装等物品"
            ]
        )
    ]
    
    # Create the page containing all questions
    page = Page(
        name="low_carbon_awareness",
        elements=questions
    )
    
    # Create the complete survey
    survey = Survey(
        id=survey_id,
        title="城市居民生活调查",
        description="本调查旨在评估城市居民的日常生活习惯",
        pages=[page],
        created_at=datetime.now()
    )
    
    return survey

SCORE_MAP = {'A': 100, 'B': 75, 'C': 50, 'D': 25}

def extract_survey_scores(responses: list[Any]) -> list[float]:
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
    score_number = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for answer in responses:
        try:
            result_dict = json.loads(answer)
            answers = list(result_dict.values())
            total_score = 0
            for ans in answers:
                if 'A' in ans or 'a' in ans:
                    score_number['A'] += 1
                    total_score += 100
                elif 'B' in ans or 'b' in ans:
                    score_number['B'] += 1
                    total_score += 75
                elif 'C' in ans or 'c' in ans:
                    score_number['C'] += 1
                    total_score += 50
                elif 'D' in ans or 'd' in ans:
                    score_number['D'] += 1
                    total_score += 25
            count = len(answers)
            score = total_score / count if count > 0 else 50
            scores.append(score)
        except Exception as e:
            print(f"error parsing survey result: {e}")
    final_score = sum(scores) / len(scores) if scores else 50

    return scores, final_score
