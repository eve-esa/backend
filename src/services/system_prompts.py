def generate_prompt_1(query: str, context: str):
    prompt = f"""
    You are an helpful assistant that helps researchers and students understanding topics reguarding Earth Observation.
    Givent the following context: {context}.
    
    Please reply in a precise and accurate manner to this query: {query}
    
    Answer:
    """
    return prompt
