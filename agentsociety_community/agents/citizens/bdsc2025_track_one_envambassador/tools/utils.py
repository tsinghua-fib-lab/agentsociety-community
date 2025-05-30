def clean_json_response(response: str) -> str:
    """remove the special characters in the response"""
    response = response.replace("```json", "").replace("```", "")
    return response.strip()