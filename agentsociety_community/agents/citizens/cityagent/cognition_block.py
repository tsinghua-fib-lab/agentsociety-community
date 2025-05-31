import logging
from typing import Optional
import jsonc
from pydantic import Field
from agentsociety.environment import Environment
from agentsociety.llm import LLM
from agentsociety.logger import get_logger
from agentsociety.memory import Memory
from agentsociety.agent import Block, FormatPrompt, BlockParams

__all__ = ["CognitionBlock"]


def extract_json(output_str):
    """Extract JSON substring from a raw string response.

    Args:
        output_str: Raw string output that may contain JSON data.

    Returns:
        Extracted JSON string if valid, otherwise None.

    Note:
        Searches for the first '{' and last '}' to isolate JSON content.
        Catches JSON decoding errors and logs warnings.
    """
    try:
        # Find the positions of the first '{' and the last '}'
        start = output_str.find("{")
        end = output_str.rfind("}")

        # Extract the substring containing the JSON
        json_str = output_str[start : end + 1]

        # Convert the JSON string to a dictionary
        return json_str
    except (ValueError, jsonc.JSONDecodeError) as e:
        get_logger().warning(f"Failed to extract JSON: {e}")
        return None


class CognitionBlockParams(BlockParams):
    top_k: int = Field(
        default=20, description="Number of most relevant memories to return"
    )


class CognitionBlock(Block):
    """A cognitive processing block handling daily updates of attitudes, thoughts, and emotions.

    Attributes:
        configurable_fields: List of configurable parameters (top_k).
        default_values: Default values for configurable parameters.
        fields_description: Metadata descriptions for configurable parameters.
        top_k: Number of most relevant memories retrieved for processing.
        last_check_time: Timestamp tracker for daily update cycles.
    """

    ParamsType = CognitionBlockParams
    name = "CognitionBlock"
    description = "Handles daily updates of attitudes, thoughts, and emotions"
    actions = {}

    def __init__(
        self,
        llm: LLM,
        environment: Environment,
        agent_memory: Memory,
        block_params: Optional[CognitionBlockParams] = None,
    ):
        """Initialize CognitionBlock with dependencies.

        Args:
            llm: Language Model interface for cognitive processing.
            environment: Environment for time-based operations.
            memory: Memory system to store/retrieve agent status and experiences.
        """
        super().__init__(
            llm=llm,
            environment=environment,
            agent_memory=agent_memory,
            block_params=block_params,
        )
        self.last_check_day = 0

    async def set_status(self, status):
        """Update multiple status fields in memory.

        Args:
            status: Dictionary of key-value pairs to update.
        """
        for key in status:
            await self.memory.status.update(key, status[key])
        return

    async def attitude_update(self):
        """Update agent's attitudes toward specific topics based on daily experiences.

        Workflow:
        1. Fetch agent's profile and current emotional state from memory.
        2. Retrieve relevant incidents using topic-based memory search.
        3. Construct a structured prompt combining profile, incidents, and previous attitude.
        4. Query LLM to generate updated attitude scores (0-10 scale).
        5. Retry up to 10 times on LLM failures.
        6. Persist updated attitudes to memory.

        Raises:
            Exception: If all LLM retries fail.
        """
        attitude = await self.memory.status.get("attitude")
        prompt_data = {
            "gender": await self.memory.status.get("gender"),
            "age": await self.memory.status.get("age"),
            "race": await self.memory.status.get("race"),
            "religion": await self.memory.status.get("religion"),
            "marriage_status": await self.memory.status.get("marriage_status"),
            "residence": await self.memory.status.get("residence"),
            "occupation": await self.memory.status.get("occupation"),
            "education": await self.memory.status.get("education"),
            "personality": await self.memory.status.get("personality"),
            "consumption": await self.memory.status.get("consumption"),
            "family_consumption": await self.memory.status.get("family_consumption"),
            "income": await self.memory.status.get("income"),
            "skill": await self.memory.status.get("skill"),
            "thought": await self.memory.status.get("thought"),
            "emotion_types": await self.memory.status.get("emotion_types"),
        }
        for topic in attitude:
            description_prompt = """
            You are a {gender}, aged {age}, belonging to the {race} race and identifying as {religion}. 
            Your marital status is {marriage_status}, and you currently reside in a {residence} area. 
            Your occupation is {occupation}, and your education level is {education}. 
            You are {personality}, with a consumption level of {consumption} and a family consumption level of {family_consumption}. 
            Your income is {income}, and you are skilled in {skill}.
            My current emotion intensities are (0 meaning not at all, 10 meaning very much):
            sadness: {sadness}, joy: {joy}, fear: {fear}, disgust: {disgust}, anger: {anger}, surprise: {surprise}.
            You have the following thoughts: {thought}.
            In the following 21 words, I have chosen {emotion_types} to represent your current status:
            Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate.
            """
            incident_str = await self.memory.stream.search(
                query=topic, top_k=self.params.top_k
            )
            if incident_str:
                incident_prompt = "Today, these incidents happened:"
                incident_prompt += incident_str
            else:
                incident_prompt = "No incidents happened today."
            previous_attitude = str(attitude[topic])  # Convert to string
            problem_prompt = (
                f"You need to decide your attitude towards topic: {topic}, "
                f"which you previously rated your attitude towards this topic as: {previous_attitude} "
                "(0 meaning oppose, 10 meaning support). "
                'Please return a new attitude rating (0-10, smaller meaning oppose, larger meaning support) in JSON format, and explain, e.g. {{"attitude": 5}}'
            )
            question_prompt = description_prompt + incident_prompt + problem_prompt
            question_prompt = FormatPrompt(question_prompt)
            emotion = await self.memory.status.get("emotion")
            sadness = emotion["sadness"]
            joy = emotion["joy"]
            fear = emotion["fear"]
            disgust = emotion["disgust"]
            anger = emotion["anger"]
            surprise = emotion["surprise"]
            prompt_data["sadness"] = sadness
            prompt_data["joy"] = joy
            prompt_data["fear"] = fear
            prompt_data["disgust"] = disgust
            prompt_data["anger"] = anger
            prompt_data["surprise"] = surprise

            await question_prompt.format(**prompt_data)
            evaluation = True
            response: dict = {}
            for retry in range(10):
                try:
                    _response = await self.llm.atext_request(
                        question_prompt.to_dialog(),
                        timeout=300,
                        response_format={"type": "json_object"},
                    )
                    json_str = extract_json(_response)
                    if json_str:
                        response = jsonc.loads(json_str)
                        evaluation = False
                        break
                except:
                    pass
            if evaluation:
                raise Exception(f"Request for attitude:{topic} update failed")
            attitude[topic] = response["attitude"]
        await self.memory.status.update("attitude", attitude)

    async def thought_update(self):
        """Generate daily reflections based on experiences and emotional state.

        Workflow:
        1. Build profile and emotion context from memory.
        2. Retrieve today's incidents.
        3. Construct a reflection prompt.
        4. Query LLM to generate thought summary and emotional keyword.
        5. Retry up to 10 times on LLM failures.
        6. Update memory with new thought and log cognition.

        Returns:
            Generated thought string.

        Raises:
            Exception: If all LLM retries fail.
        """
        description_prompt = """
        You are a {gender}, aged {age}, belonging to the {race} race and identifying as {religion}. 
        Your marital status is {marriage_status}, and you currently reside in a {residence} area. 
        Your occupation is {occupation}, and your education level is {education}. 
        You are {personality}, with a consumption level of {consumption} and a family consumption level of {family_consumption}. 
        Your income is {income}, and you are skilled in {skill}.
        My current emotion intensities are (0 meaning not at all, 10 meaning very much):
        sadness: {sadness}, joy: {joy}, fear: {fear}, disgust: {disgust}, anger: {anger}, surprise: {surprise}.
        You have the following thoughts: {thought}.
        In the following 21 words, I have chosen {emotion_types} to represent your current status:
        Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate.
        """
        incident_str = await self.memory.stream.search_today(top_k=20)
        if incident_str:
            incident_prompt = "Today, these incidents happened:\n" + incident_str
        else:
            incident_prompt = "No incidents happened today."
        question_prompt = """
            Please review what happened today and share your thoughts and feelings about it.
            Consider your current emotional state and experiences, then:
            1. Summarize your thoughts and reflections on today's events
            2. Choose one word that best describes your current emotional state from: Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate.
            Return in JSON format, e.g. {{"thought": "Currently nothing good or bad is happening, I think ...."}}"""
        question_prompt = description_prompt + incident_prompt + question_prompt
        question_prompt = FormatPrompt(question_prompt)
        emotion = await self.memory.status.get("emotion")
        sadness = emotion["sadness"]
        joy = emotion["joy"]
        fear = emotion["fear"]
        disgust = emotion["disgust"]
        anger = emotion["anger"]
        surprise = emotion["surprise"]
        await question_prompt.format(
            gender=await self.memory.status.get("gender"),
            age=await self.memory.status.get("age"),
            race=await self.memory.status.get("race"),
            religion=await self.memory.status.get("religion"),
            marriage_status=await self.memory.status.get("marriage_status"),
            residence=await self.memory.status.get("residence"),
            occupation=await self.memory.status.get("occupation"),
            education=await self.memory.status.get("education"),
            personality=await self.memory.status.get("personality"),
            consumption=await self.memory.status.get("consumption"),
            family_consumption=await self.memory.status.get("family_consumption"),
            income=await self.memory.status.get("income"),
            skill=await self.memory.status.get("skill"),
            sadness=sadness,
            joy=joy,
            fear=fear,
            disgust=disgust,
            anger=anger,
            surprise=surprise,
            emotion=await self.memory.status.get("emotion"),
            thought=await self.memory.status.get("thought"),
            emotion_types=await self.memory.status.get("emotion_types"),
        )

        evaluation = True
        response: dict = {}
        for retry in range(10):
            try:
                _response = await self.llm.atext_request(
                    question_prompt.to_dialog(),
                    timeout=300,
                    response_format={"type": "json_object"},
                )
                json_str = extract_json(_response)
                if json_str:
                    response = jsonc.loads(json_str)
                    evaluation = False
                    break
            except:
                pass
        if evaluation:
            raise Exception("Request for cognition update failed")

        thought = str(response["thought"])
        await self.memory.status.update("thought", thought)
        await self.memory.stream.add_cognition(description=thought)

        return thought

    async def cross_day(self):
        """Check if a new day has started in the simulation environment.

        Returns:
            True if a new day is detected, False otherwise.
        """
        day, _ = self.environment.get_datetime()
        if self.last_check_day == 0:
            self.last_check_day = day
            return False
        if day > self.last_check_day:
            self.last_check_day = day
            return True
        return False

    async def forward(self):
        """Main daily cognitive update entry point.

        Triggers:
            - thought_update()
            - attitude_update()
        Only executes when cross_day() detects a new day.
        """
        # cognition update: thought and attitude
        if await self.cross_day():
            await self.thought_update()
            await self.attitude_update()

    async def emotion_update(self, incident):
        """Update emotion intensities based on a specific incident.

        Args:
            incident: Description of the triggering event.

        Returns:
            Natural language conclusion about emotional state.

        Raises:
            Exception: If LLM requests fail after 10 retries.

        Workflow:
            1. Build emotion context from current state
            2. Incorporate incident details into prompt
            3. Query LLM for updated emotion scores and summary
            4. Update memory with new emotional state
        """
        description_prompt = """
        You are a {gender}, aged {age}, belonging to the {race} race and identifying as {religion}. 
        Your marital status is {marriage_status}, and you currently reside in a {residence} area. 
        Your occupation is {occupation}, and your education level is {education}. 
        You are {personality}, with a consumption level of {consumption} and a family consumption level of {family_consumption}. 
        Your income is {income}, and you are skilled in {skill}.
        My current emotion intensities are (0 meaning not at all, 10 meaning very much):
        sadness: {sadness}, joy: {joy}, fear: {fear}, disgust: {disgust}, anger: {anger}, surprise: {surprise}.
        You have the following thoughts: {thought}.
        In the following 21 words, choose one word to represent your current status:
        [Joy, Distress, Resentment, Pity, Hope, Fear, Satisfaction, Relief, Disappointment, Pride, Admiration, Shame, Reproach, Liking, Disliking, Gratitude, Anger, Gratification, Remorse, Love, Hate].
        """

        incident_prompt = f"{incident}"  # waiting for incident port
        question_prompt = """
            Please reconsider your emotion intensities: 
            sadness, joy, fear, disgust, anger, surprise (0 meaning not at all, 10 meaning very much).
            Return in JSON format, e.g. {{"sadness": 5, "joy": 5, "fear": 5, "disgust": 5, "anger": 5, "surprise": 5, "conclusion": "I feel ...", "word": "Relief"}}"""
        question_prompt = description_prompt + incident_prompt + question_prompt
        question_prompt = FormatPrompt(question_prompt)
        emotion = await self.memory.status.get("emotion")
        sadness = emotion["sadness"]
        joy = emotion["joy"]
        fear = emotion["fear"]
        disgust = emotion["disgust"]
        anger = emotion["anger"]
        surprise = emotion["surprise"]
        await question_prompt.format(
            gender=await self.memory.status.get("gender"),
            age=await self.memory.status.get("age"),
            race=await self.memory.status.get("race"),
            religion=await self.memory.status.get("religion"),
            marriage_status=await self.memory.status.get("marriage_status"),
            residence=await self.memory.status.get("residence"),
            occupation=await self.memory.status.get("occupation"),
            education=await self.memory.status.get("education"),
            personality=await self.memory.status.get("personality"),
            consumption=await self.memory.status.get("consumption"),
            family_consumption=await self.memory.status.get("family_consumption"),
            income=await self.memory.status.get("income"),
            skill=await self.memory.status.get("skill"),
            sadness=sadness,
            joy=joy,
            fear=fear,
            disgust=disgust,
            anger=anger,
            surprise=surprise,
            emotion=await self.memory.status.get("emotion"),
            thought=await self.memory.status.get("thought"),
            emotion_types=await self.memory.status.get("emotion_types"),
        )

        evaluation = True
        response: dict = {}
        for retry in range(10):
            try:
                _response = await self.llm.atext_request(
                    question_prompt.to_dialog(),
                    timeout=300,
                    response_format={"type": "json_object"},
                )
                json_str = extract_json(_response)
                if json_str:
                    response = jsonc.loads(json_str)
                    evaluation = False
                    break
            except Exception as e:
                get_logger().warning(f"Request for cognition update failed: {e}")
                pass
        if evaluation:
            raise Exception("Request for cognition update failed")

        await self.memory.status.update(
            "emotion",
            {
                "sadness": int(response["sadness"]),
                "joy": int(response["joy"]),
                "fear": int(response["fear"]),
                "disgust": int(response["disgust"]),
                "anger": int(response["anger"]),
                "surprise": int(response["surprise"]),
            },
        )
        await self.memory.status.update("emotion_types", str(response["word"]))
        return response["conclusion"]
