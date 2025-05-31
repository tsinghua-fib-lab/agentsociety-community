import numbers
import numpy as np

from agentsociety.agent import Block
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from .utils import *


class MonthEconomyPlanBlock(Block):
    """Manages monthly economic planning and mental health assessment.

    Attributes:
        configurable_fields: Economic policy parameters
        economy_client: Interface to economic system
        llm_error: Counter for LLM failures
    """

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        ubi: float = 0,
        num_labor_hours: int = 168,
        productivity_per_labor: float = 1,
        time_diff: int = 30 * 24 * 60 * 60,
    ):
        super().__init__(llm=llm, environment=environment, agent_memory=agent_memory)
        self.llm_error = 0
        self.last_time_trigger = None
        self.forward_times = 0
        self.ubi = ubi
        self.num_labor_hours = num_labor_hours
        self.productivity_per_labor = productivity_per_labor
        self.time_diff = time_diff

    async def month_trigger(self):
        """Check if monthly planning cycle should activate."""
        now_tick = self.environment.get_tick()
        if (
            self.last_time_trigger is None
            or now_tick - self.last_time_trigger >= self.time_diff
        ):
            self.last_time_trigger = now_tick
            return True
        return False

    async def forward(self):
        """Execute monthly planning workflow.

        Workflow:
            1. Collect economic indicators
            2. Generate LLM prompts for work/consumption propensity
            3. Update agent's economic status
            4. Periodically conduct mental health assessments
            5. Handle UBI policy evaluations
        """
        if await self.month_trigger():
            agent_id = await self.memory.status.get("id")
            firms_id = await self.environment.economy_client.get_firm_ids()
            firm_id = await self.memory.status.get("firm_id")
            bank_id = await self.environment.economy_client.get_bank_ids()
            bank_id = bank_id[0]
            name = await self.memory.status.get("name")
            age = await self.memory.status.get("age")
            city = await self.memory.status.get("city")
            job = await self.memory.status.get("occupation")
            skill, consumption, wealth = await self.environment.economy_client.get(
                agent_id, ["skill", "consumption", "currency"]
            )
            get_logger().debug(f"type of skill: {type(skill)}, value: {skill}")
            tax_paid = await self.memory.status.get("tax_paid")
            prices = await self.environment.economy_client.get(firms_id, "price")
            price = np.mean(prices)
            interest_rate = await self.environment.economy_client.get(
                bank_id, "interest_rate"
            )

            problem_prompt = f"""
                    You're {name}, a {age}-year-old individual living in {city}. As with all Americans, a portion of your monthly income is taxed by the federal government. This taxation system is tiered, income is taxed cumulatively within defined brackets, combined with a redistributive policy: after collection, the government evenly redistributes the tax revenue back to all citizens, irrespective of their earnings.
                """
            job_prompt = f"""
                        In the previous month, you worked as a(an) {job}. If you continue working this month, your expected hourly income will be ${skill:.2f}.
                    """
            consumption_propensity = await self.memory.status.get(
                "consumption_propensity"
            )
            if (consumption <= 0) and (consumption_propensity > 0):
                consumption_prompt = f"""
                            Besides, you had no consumption due to shortage of goods.
                        """
            else:
                consumption_prompt = f"""
                            Besides, your consumption was ${consumption:.2f}.
                        """
            tax_prompt = f"""Your tax deduction amounted to ${tax_paid:.2f}, and the government uses the tax revenue to provide social services to all citizens."""
            if self.ubi and self.forward_times >= 96:
                tax_prompt = f"{tax_prompt} Specifically, the government directly provides ${self.ubi} per capita in each month."
            price_prompt = f"""Meanwhile, in the consumption market, the average price of essential goods is now at ${price:.2f}."""
            job_prompt = prettify_document(job_prompt)
            obs_prompt = f"""
                            {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                            Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                            Your goal is to maximize your utility by deciding how much to work and how much to consume. Your utility is determined by your consumption, income, saving, social service recieved and leisure time. You will spend the time you do not work on leisure activities. 
                            With all these factors in play, and considering aspects like your living costs, any future aspirations, and the broader economic trends, how is your willingness to work this month? Furthermore, how would you plan your expenditures on essential goods, keeping in mind good price?
                            Please share your decisions in a JSON format as follows:
                            {{'work': a value between 0 and 1, indicating the propensity to work,
                            'consumption': a value between 0 and 1, indicating the proportion of all your savings and income you intend to spend on essential goods
                            }}
                            Any other output words are NOT allowed.
                        """
            obs_prompt = prettify_document(obs_prompt)
            try:
                await self.memory.status.update(
                    "dialog_queue",
                    [{"role": "user", "content": obs_prompt}],
                    mode="merge",
                )
                dialog_queue = await self.memory.status.get("dialog_queue")
                content = await self.llm.atext_request(list(dialog_queue), timeout=300)
                await self.memory.status.update(
                    "dialog_queue",
                    [{"role": "assistant", "content": content}],
                    mode="merge",
                )
                propensity_dict = extract_dict_from_string(content)[0]
                work_propensity, consumption_propensity = (
                    propensity_dict["work"],
                    propensity_dict["consumption"],
                )
                if isinstance(work_propensity, numbers.Number) and isinstance(
                    consumption_propensity, numbers.Number
                ):
                    await self.memory.status.update("work_propensity", work_propensity)
                    await self.memory.status.update(
                        "consumption_propensity", consumption_propensity
                    )
                else:
                    self.llm_error += 1
            except:
                self.llm_error += 1

            work_skill = await self.environment.economy_client.get(agent_id, "skill")
            work_propensity = await self.memory.status.get("work_propensity")
            consumption_propensity = await self.memory.status.get(
                "consumption_propensity"
            )
            work_hours = work_propensity * self.num_labor_hours
            # income = await self.economy_client.get(agent_id, 'income')
            income = work_hours * work_skill

            wealth = await self.environment.economy_client.get(agent_id, "currency")
            wealth += work_hours * work_skill
            await self.environment.economy_client.update(agent_id, "currency", wealth)
            await self.environment.economy_client.delta_update_firms(
                firm_id, delta_inventory=int(work_hours * self.productivity_per_labor)
            )

            if self.ubi and self.forward_times >= 96:
                income += self.ubi
                wealth += self.ubi

            await self.memory.status.update(
                "to_consumption_currency", consumption_propensity * wealth
            )

            await self.environment.economy_client.update(agent_id, "consumption", 0)
            await self.environment.economy_client.update(agent_id, "income", income)
            await self.environment.economy_client.update(agent_id, "currency", wealth)

            if self.forward_times % 3 == 0:
                obs_prompt = f"""
                                {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                                Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                                Please fill in the following questionnaire:
                                Indicate how often you have felt this way during the last week by choosing one of the following options:
                                "Rarely" means Rarely or none of the time (less than 1 day),
                                "Some" means Some or a little of the time (1-2 days),
                                "Occasionally" means Occasionally or a moderate amount of the time (3-4 days),
                                "Most" means Most or all of the time (5-7 days).
                                Statement 1: I was bothered by things that usually don't bother me.  
                                Statement 2: I did not feel like eating; my appetite was poor.
                                Statement 3: I felt that I could not shake off the blues even with help from my family or friends.
                                Statement 4: I felt that I was just as good as other people.
                                Statement 5: I had trouble keeping my mind on what I was doing.
                                Statement 6: I felt depressed.
                                Statement 7: I felt that everything I did was an effort.
                                Statement 8: I felt hopeful about the future.
                                Statement 9: I thought my life had been a failure.
                                Statement 10: I felt fearful.
                                Statement 11: My sleep was restless.
                                Statement 12: I was happy.
                                Statement 13: I talked less than usual.
                                Statement 14: I felt lonely.
                                Statement 15: People were unfriendly.
                                Statement 16: I enjoyed life.
                                Statement 17: I had crying spells.
                                Statement 18: I felt sad.
                                Statement 19: I felt that people disliked me.
                                Statement 20: I could not get "going".
                                Please response with json format with keys being numbers 1-20 and values being one of "Rarely", "Some", "Occasionally", "Most".
                                Any other output words are NOT allowed.
                            """
                obs_prompt = prettify_document(obs_prompt)
                content = await self.llm.atext_request(
                    [{"role": "user", "content": obs_prompt}], timeout=300
                )
                inverse_score_items = [3, 8, 12, 16]
                category2score = {"rarely": 0, "some": 1, "occasionally": 2, "most": 3}
                try:
                    content = extract_dict_from_string(content)[0]
                    for k in content:
                        if k in inverse_score_items:
                            content[k] = 3 - category2score[content[k].lower()]
                        else:
                            content[k] = category2score[content[k].lower()]
                    depression = sum(list(content.values()))
                    await self.memory.status.update("depression", depression)
                except:
                    self.llm_error += 1

            if self.ubi and self.forward_times >= 96 and self.forward_times % 12 == 0:
                obs_prompt = f"""
                                {problem_prompt} {job_prompt} {consumption_prompt} {tax_prompt} {price_prompt}
                                Your current savings account balance is ${wealth:.2f}. Interest rates, as set by your bank, stand at {interest_rate*100:.2f}%. 
                                What's your opinion on the UBI policy, including the advantages and disadvantages?
                            """
                obs_prompt = prettify_document(obs_prompt)
                content = await self.llm.atext_request(
                    [{"role": "user", "content": obs_prompt}], timeout=300
                )
                await self.memory.status.update("ubi_opinion", [content], mode="merge")

            self.forward_times += 1
