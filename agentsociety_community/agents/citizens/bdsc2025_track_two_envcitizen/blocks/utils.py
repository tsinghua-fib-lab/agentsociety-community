import ast
import re

TIME_ESTIMATE_PROMPT = """As an intelligent agent's time estimation system, please estimate the time needed to complete the current action based on the overall plan and current intention.

Overall plan:
{plan}

Current action: {intention}

Current emotion: {emotion_types}

Examples:
- "Learn programming": {{"time": 120}}
- "Watch a movie": {{"time": 150}} 
- "Play mobile games": {{"time": 60}}
- "Read a book": {{"time": 90}}
- "Exercise": {{"time": 45}}

Please return the result in JSON format (Do not return any other text), the time unit is [minute], example:
{{
    "time": 10
}}
"""


def prettify_document(document: str) -> str:
    # Remove sequences of whitespace characters (including newlines)
    cleaned = re.sub(r"\s+", " ", document).strip()
    return cleaned


def extract_dict_from_string(input_string):
    """
    Extract dictionaries from the input string. Supports multi-line dictionaries and nested dictionaries.
    """
    # Use regular expression to find all possible dictionary parts, allowing multi-line dictionaries
    dict_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"  # Regular expression to match dictionaries, supports nesting
    matches = re.findall(
        dict_pattern, input_string, re.DOTALL
    )  # re.DOTALL allows matching newline characters

    dicts = []

    for match in matches:
        try:
            # Use ast.literal_eval to convert the string to a dictionary
            parsed_dict = ast.literal_eval(match)
            if isinstance(parsed_dict, dict):
                dicts.append(parsed_dict)
        except (ValueError, SyntaxError) as e:
            print(f"Failed to parse dictionary: {e}")

    return dicts


def clean_json_response(response: str) -> str:
    """remove the special characters in the response"""
    response = response.replace("```json", "").replace("```", "")
    return response.strip()
